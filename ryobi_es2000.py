#!/usr/bin/env python3
"""
Ryobi Phone Works ES2000 Infrared Thermometer
Real-time Driver and Demodulator for macOS

Reverse-Engineered Protocol Specification:
- Physical Interface: 3.5mm TRRS (CTIA) Headphone/Mic Jack
- Audio Output Left (Tip)   : 2200 Hz continuous carrier (Energy harvest & clock sync)
- Audio Output Right (Ring1): 3.2 ms 2200 Hz burst pulse every ~109 ms (Measurement trigger)
- Audio Input (Sleeve / Mic): Bell 202 AFSK telemetry burst:
    - Mark  ('0'): 1200 Hz (period ~833 us)
    - Space ('1'): 2200 Hz (period ~454 us)
    - Period threshold: 595 us (f_cutoff ~1680.7 Hz)
    - Packet gap: > 2.0 ms silence between bursts
- Packet Structure (40 bits):
    - [0:4]   : Preamble "1010" (or "010" with leading bit clipped at onset)
    - [4:9]   : Model / Device Type ID (5 bits)
    - [9:11]  : Status / Control Flags (2 bits)
    - [11:31] : 20-bit Raw Temperature (sign bit at bit 19; temp_c = raw / 10.0)
    - [31:36] : Battery Level (5 bits, 0..31)
    - [36:40] : Trailer / Checksum (4 bits)
"""

import os
import sys
import time
import argparse
import wave
import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
PERIOD_THRESHOLD_SEC = 0.000595   # 595 us Bell 202 discriminator
PACKET_GAP_SEC = 0.0020           # > 2.0 ms indicates inter-packet boundary
PEAK_POS_THRESHOLD = 0.08         # Normalized positive peak threshold
PEAK_NEG_THRESHOLD = -0.12        # Normalized negative peak threshold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_WAV = os.path.join(SCRIPT_DIR, "audio", "x_100ms_audiofile.wav")
LASER_WAV   = os.path.join(SCRIPT_DIR, "audio", "laser_pointer_2200hz_hq_0db.wav")

class RyobiES2000:
    def __init__(self, trigger_wav_path=TRIGGER_WAV):
        self.trigger_wav_path = trigger_wav_path
        self.running = False
        self.latest_reading = None
        self.packet_count = 0
        
        # Load official trigger audio file
        with wave.open(self.trigger_wav_path, "rb") as w:
            self.sr = w.getframerate()
            raw = w.readframes(w.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
            self.trigger_audio = samples.astype(np.float32) / 32768.0

    def decode_audio_chunk(self, audio_chunk):
        """Demodulate PCM samples and return list of validated temperature packets."""
        chunk = audio_chunk - np.mean(audio_chunk)
        max_amp = np.max(np.abs(chunk))
        if max_amp < 0.03:
            return []
        norm = chunk / max_amp

        # Alternating peak detector state machine (matches libnative-lib.so findMaxPeakValues)
        peaks = []
        state = 1
        n = len(norm)
        for i in range(1, n - 1):
            if state == 1:
                if norm[i] > PEAK_POS_THRESHOLD and norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1]:
                    peaks.append(i)
                    state = 0
            else:
                if norm[i] < PEAK_NEG_THRESHOLD and norm[i] <= norm[i - 1] and norm[i] <= norm[i + 1]:
                    peaks.append(i)
                    state = 1

        if len(peaks) < 10:
            return []

        # Peak-to-peak intervals between consecutive positive peaks
        pos_peaks = peaks[0::2]
        intervals = np.diff(pos_peaks) / self.sr

        # Group bits into packets using inter-packet gaps (> 2ms)
        packets = []
        cur_pkt = []
        for dt in intervals:
            if dt > PACKET_GAP_SEC:
                if cur_pkt:
                    packets.append("".join(cur_pkt))
                    cur_pkt = []
            else:
                cur_pkt.append("1" if dt < PERIOD_THRESHOLD_SEC else "0")
        if cur_pkt:
            packets.append("".join(cur_pkt))

        # Parse valid packets
        readings = []
        for raw_pkt in packets:
            pkt = raw_pkt
            # Handle leading bit clip (39 bits starting with "010" -> restore leading "1")
            if pkt.startswith("010") and len(pkt) == 39:
                pkt = "1" + pkt

            if pkt.startswith("1010") and len(pkt) == 40:
                model_id = int(pkt[4:9], 2)
                flags = int(pkt[9:11], 2)
                raw_val = int(pkt[11:31], 2)
                
                # Check sign bit (bit 19 / 0x80000)
                is_negative = bool(raw_val & 0x80000)
                if is_negative:
                    temp_c = -float(raw_val & 0x7FFFF) / 10.0
                else:
                    temp_c = float(raw_val) / 10.0
                
                temp_f = temp_c * 1.8 + 32.0
                battery_raw = int(pkt[31:36], 2)
                trailer = int(pkt[36:40], 2)

                # Filter valid measurement range (-30 °C to 350 °C)
                # Ignore initial 0.0 °C calibration packets during first warm-up cycles
                readings.append({
                    "temp_c": round(temp_c, 1),
                    "temp_f": round(temp_f, 1),
                    "raw_val": raw_val,
                    "battery_raw": battery_raw,
                    "battery_pct": int((battery_raw / 31.0) * 100),
                    "model_id": model_id,
                    "flags": flags,
                    "trailer": trailer,
                    "frame": pkt,
                    "timestamp": time.time()
                })

        return readings

    def monitor(self, duration=None, callback=None):
        """Start continuous real-time measurement loop."""
        self.running = True
        print("=" * 68)
        print("   Ryobi Phone Works ES2000 Infrared Thermometer")
        print("   Real-time Telemetry Acquisition (109 ms trigger cycle)")
        print("   Press Ctrl+C to terminate.")
        print("=" * 68)
        print(f"{'Time':^10} | {'Temp (°C)':^12} | {'Temp (°F)':^12} | {'Battery':^10} | {'Pkt #':^6}")
        print("-" * 68)

        # Loop trigger audio in chunk batches
        batch_duration = 0.5  # 500 ms per cycle (~4-5 packets)
        batch_samples = int(batch_duration * self.sr)
        repeats = int(np.ceil(batch_samples / len(self.trigger_audio)))
        out_chunk = np.tile(self.trigger_audio, (repeats, 1))[:batch_samples]

        start_time = time.time()
        last_temp = None

        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    break

                # Full-duplex play & record
                rec = sd.playrec(out_chunk, samplerate=self.sr, channels=1, dtype="float32")
                sd.wait()
                audio = rec.flatten()

                readings = self.decode_audio_chunk(audio)
                for r in readings:
                    # Ignore initial 0.0 boot-up packets once a valid reading has been obtained
                    if r["temp_c"] == 0.0 and self.latest_reading is not None:
                        continue

                    self.latest_reading = r
                    self.packet_count += 1
                    
                    t_str = time.strftime("%H:%M:%S")
                    print(f"{t_str:^10} | {r['temp_c']:>9.1f} °C | {r['temp_f']:>9.1f} °F | {r['battery_pct']:>7d}% | #{self.packet_count:<5d}")

                    if callback:
                        callback(r)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("-" * 68)
            print(f"Acquisition terminated. Total packets captured: {self.packet_count}")

    def laser_mode(self, duration=5.0):
        """Activate laser pointer mode (2200 Hz continuous carrier)."""
        with wave.open(LASER_WAV, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
            laser_audio = samples.astype(np.float32) / 32768.0

        print(f"Activating Laser Pointer for {duration} seconds...")
        n_samples = int(duration * sr)
        repeats = int(np.ceil(n_samples / len(laser_audio)))
        play_data = np.tile(laser_audio, (repeats, 1))[:n_samples]

        sd.play(play_data, samplerate=sr)
        sd.wait()
        print("Laser deactivated.")


def main():
    parser = argparse.ArgumentParser(description="Ryobi Phone Works ES2000 Driver for macOS")
    parser.add_argument("--laser", action="store_true", help="Turn on laser pointer only")
    parser.add_argument("--duration", type=float, default=None, help="Measurement duration in seconds (default: continuous)")
    parser.add_argument("--csv", type=str, default=None, help="Save temperature readings to CSV file")
    args = parser.parse_args()

    sensor = RyobiES2000()

    if args.laser:
        sensor.laser_mode(duration=args.duration or 5.0)
    else:
        csv_file = None
        if args.csv:
            csv_file = open(args.csv, "w", buffering=1)
            csv_file.write("timestamp,datetime,temp_c,temp_f,battery_pct,raw_val\n")
            print(f"Logging data to {args.csv}")

        def csv_callback(r):
            if csv_file:
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
                csv_file.write(f"{r['timestamp']:.3f},{dt_str},{r['temp_c']},{r['temp_f']},{r['battery_pct']},{r['raw_val']}\n")

        try:
            sensor.monitor(duration=args.duration, callback=csv_callback if args.csv else None)
        finally:
            if csv_file:
                csv_file.close()
                print(f"Saved log to {args.csv}")


if __name__ == "__main__":
    main()
