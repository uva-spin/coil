#!/usr/bin/env python3
"""tinySA serial I/O: read power per frequency bin (dBm and mW).
Library:
    with TinySA() as sa:
        bins = sa.read(center_hz=213e6, span_hz=400e3, n=500, rbw_hz=10e3)
        # -> list of (bin, freq_hz, dbm, mw)
CLI:
    python tinySA_IO.py            # uses the defaults below
    python tinySA_IO.py --center 213e6 --span 400e3 --bins 500 --rbw 10e3 --out bins.csv   # all optional
"""
import argparse
import csv
import re
import time

import numpy as np
import serial
import serial.tools.list_ports

VID = 0x0483        # STMicroelectronics, the tinySA's USB vendor ID
CENTER_HZ = 213e6
SPAN_HZ = 400e3
BINS = 500
RBW_HZ = 1e3
OUT = "bins.csv"

MAX_SCAN = 250     # the tinySA rejects large scans, so longer grids are stitched


def find_ports():
    """Serial ports that look like a tinySA, best match first (macOS, Windows and Linux)."""
    ranked = []
    for p in serial.tools.list_ports.comports():
        text = f"{p.description} {p.manufacturer} {p.product}".lower()
        if p.vid == VID:
            rank = 0
        elif "tinysa" in text or "stm" in text:
            rank = 1
        elif "usbmodem" in p.device or "ttyACM" in p.device:
            rank = 2
        else:
            continue
        ranked.append((rank, p.device))
    return [dev for _, dev in sorted(ranked)]




def _number(token):
    """The firmware prints exact powers of ten with ':' as the leading digit (':.0e-01' is 1.0)."""
    try:
        return float(token)
    except ValueError:
        return float(re.sub(r"^([+-]?):\.", r"\g<1>10.", token))


class TinySA:
    def __init__(self, port=None, tries=5):
        candidates = [port] if port else find_ports()
        if not candidates:
            seen = ", ".join(f"{p.device} ({p.description})" for p in serial.tools.list_ports.comports()) or "none"
            raise RuntimeError(f"No tinySA found (serial ports seen: {seen}). "
                               "Is it plugged in with a data cable? Try --port COMx.")
        err = None
        for _ in range(tries):  # the first open often fails right after enumeration
            for dev in candidates:
                self.ser = None
                try:
                    self.ser = serial.Serial(dev, 115200, timeout=0.5)
                    time.sleep(0.3)
                    self.cmd("version", 3)
                    self.cmd("trace dBm", 3)  # scan units follow the display unit, so force dBm
                    self.cmd("pause")
                    return
                except Exception as e:
                    err = e
                    if self.ser:
                        self.ser.close()  # Windows ports are exclusive, so release before retrying
            time.sleep(1)
        raise RuntimeError(f"Could not open the tinySA on {', '.join(candidates)}: {err}")

    def cmd(self, text, timeout=15.0):
        """Send a command, return its response lines."""
        self.ser.reset_input_buffer()
        self.ser.write(text.encode() + b"\r")
        buf, end = b"", time.monotonic() + timeout
        while not buf.rstrip().endswith(b"ch>"):
            if time.monotonic() > end:
                raise TimeoutError(f"tinySA did not answer {text!r}")
            buf += self.ser.read(4096)
        lines = buf.decode(errors="replace").splitlines()[1:]
        return [l.strip() for l in lines if l.strip() and not l.startswith("ch>")]

    def read(self, center_hz, span_hz, n=500, rbw_hz=10e3):
        """One sweep on an n-point grid centered on center_hz. Returns [(bin, freq_hz, dbm, mw)]."""
        self.cmd(f"rbw {rbw_hz / 1e3:g}")
        grid = np.linspace(center_hz - span_hz / 2, center_hz + span_hz / 2, n)
        pts = []
        for idx in np.array_split(np.arange(n), -(-n // MAX_SCAN)):
            lo, hi = int(round(grid[idx[0]])), int(round(grid[idx[-1]]))
            rows = []
            for line in self.cmd(f"scan {lo} {hi} {len(idx)} 3"):
                parts = line.split()
                try:
                    rows.append((float(parts[0]), _number(parts[1])))
                except (ValueError, IndexError):
                    pass
            if len(rows) != len(idx):
                raise RuntimeError(f"tinySA returned {len(rows)} of {len(idx)} points for {lo}-{hi} Hz")
            pts += rows
        return [(i, f, d, 10 ** (d / 10)) for i, (f, d) in enumerate(pts)]

    def close(self):
        try:
            self.cmd("resume")
        finally:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    p = argparse.ArgumentParser(description="Read power per bin from a tinySA.")
    p.add_argument("--port", help="serial port, e.g. COM3 (default: auto-detect)")
    p.add_argument("--center", type=float, default=CENTER_HZ, help=f"center frequency, Hz (default {CENTER_HZ:g})")
    p.add_argument("--span", type=float, default=SPAN_HZ, help=f"span, Hz (default {SPAN_HZ:g})")
    p.add_argument("--bins", type=int, default=BINS, help=f"number of bins (default {BINS})")
    p.add_argument("--rbw", type=float, default=RBW_HZ, help=f"RBW, Hz (default {RBW_HZ:g})")
    p.add_argument("--out", default=OUT, help=f"CSV path, empty string to skip (default {OUT})")
    a = p.parse_args()

    with TinySA(a.port) as sa:
        rows = sa.read(a.center, a.span, a.bins, a.rbw)

    print(f"center {a.center / 1e6:.4f} MHz, span {a.span / 1e3:g} kHz, {len(rows)} bins")
    print(f"{'bin':>5} {'freq_MHz':>12} {'dBm':>9} {'mW':>12}")
    for i, f, d, mw in rows:
        print(f"{i:5d} {f / 1e6:12.4f} {d:9.2f} {mw:12.4g}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "center_mhz", "freq_mhz", "dbm", "mw"])
            w.writerows([i, f"{a.center / 1e6:.6f}", f"{fr / 1e6:.6f}", f"{d:.3f}", f"{mw:.6g}"]
                        for i, fr, d, mw in rows)


if __name__ == "__main__":
    main()
