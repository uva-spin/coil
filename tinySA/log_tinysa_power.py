#!/usr/bin/env python3
import argparse
import csv
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import serial
import serial.tools.list_ports


TINYSA_VID = 0x0483  # STMicroelectronics, the USB vendor ID the tinySA reports


def find_ports():
    """Serial ports that look like a tinySA, best match first (macOS, Windows and Linux)."""
    ranked = []
    for p in serial.tools.list_ports.comports():
        text = f"{p.description} {p.manufacturer} {p.product}".lower()
        if p.vid == TINYSA_VID:
            rank = 0
        elif "tinysa" in text or "stm" in text:
            rank = 1
        elif "usbmodem" in p.device or "ttyACM" in p.device:
            rank = 2
        else:
            continue
        ranked.append((rank, p.device))
    return [dev for _, dev in sorted(ranked)]


def open_tinysa(port=None, tries=5):
    """Open the tinySA (auto-detected unless port is given) and confirm it answers.

    The first open sometimes fails right after the device enumerates, so retry a few times.
    """
    candidates = [port] if port else find_ports()
    if not candidates:
        seen = ", ".join(p.device for p in serial.tools.list_ports.comports()) or "none"
        raise RuntimeError(f"No tinySA found (serial ports seen: {seen}). Is it plugged in with a data cable?")
    last = None
    for _ in range(tries):
        for dev in candidates:
            ser = None
            try:
                ser = serial.Serial(dev, 115200, timeout=0.5)
                time.sleep(0.3)
                command(ser, "version", timeout=3.0)
                # scan results follow the instrument's display unit, so force dBm
                reply = command(ser, "trace dBm", timeout=3.0)
                if any("usage" in l.lower() for l in reply):
                    print("Warning: could not force the tinySA to dBm. Set its unit to dBm on the device.",
                          file=sys.stderr)
                return ser
            except Exception as e:
                last = e
                if ser:
                    ser.close()
        time.sleep(1.0)
    raise RuntimeError(f"Could not open the tinySA on {', '.join(candidates)}: {last}")


def command(ser, cmd, timeout=15.0):
    """Send a command and return the response lines (echo and prompt stripped)."""
    ser.reset_input_buffer()
    ser.write(cmd.encode() + b"\r")
    buf = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buf += ser.read(4096)
        if buf.rstrip().endswith(b"ch>"):
            break
    else:
        raise TimeoutError(f"tinySA did not answer {cmd!r} within {timeout}s")
    lines = buf.decode(errors="replace").splitlines()
    return [l.strip() for l in lines[1:] if l.strip() and not l.startswith("ch>")]


MAX_SCAN_POINTS = 250  # the tinySA rejects very large scans, so long grids are stitched


def _level(token):
    """Parse a tinySA level token. Exact powers of ten are printed with ':' for the leading 1."""
    try:
        return float(token)
    except ValueError:
        return float(re.sub(r"^([+-]?):\.", r"\g<1>10.", token))


def acquire(ser, start_hz, stop_hz, n):
    """Return [(freq_hz, dbm)] on an evenly spaced n-point grid from start to stop.

    Grids longer than MAX_SCAN_POINTS are measured as several consecutive scans of nearly
    equal size (no point is measured twice). Raises RuntimeError if the tinySA returns a
    partial or off-grid trace, so a bad sweep is never logged as a good one.
    Reported levels are kept as-is, including values above the analyzer's linear range.
    """
    if n < 2:
        raise ValueError("need at least 2 points per sweep")
    grid = np.linspace(start_hz, stop_hz, n)
    chunks = np.array_split(np.arange(n), -(-n // MAX_SCAN_POINTS))
    pts = []
    for idx in chunks:
        k = len(idx)
        lo, hi = int(round(grid[idx[0]])), int(round(grid[idx[-1]]))
        chunk = []
        for line in command(ser, f"scan {lo} {hi} {k} 3"):
            parts = line.split()
            print(parts)
            if len(parts) >= 2:
                try:
                    chunk.append((float(parts[0]), _level(parts[1])))
                except ValueError:
                    pass
        if len(chunk) != k:
            raise RuntimeError(f"tinySA returned {len(chunk)} of {k} points for scan {lo}-{hi} Hz")
        fr = np.array([c[0] for c in chunk])
        db = np.array([c[1] for c in chunk])
        if not np.all(np.isfinite(fr)):
            raise RuntimeError("tinySA returned non-numeric frequencies")
        if np.max(np.abs(fr - grid[idx])) > (hi - lo) / max(k - 1, 1) / 2 + 1:
            raise RuntimeError("tinySA frequency grid does not match the request")
        pts.extend(chunk)
    return pts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="Serial port, e.g. COM3 or /dev/cu.usbmodem4001 (default: auto-detect)")
    p.add_argument("--center", type=float, help="Center frequency in Hz (use with --span)")
    p.add_argument("--span", type=float, help="Span in Hz (use with --center)")
    p.add_argument("--start", type=float, help="Start frequency in Hz (alternative to --center/--span)")
    p.add_argument("--stop", type=float, help="Stop frequency in Hz (alternative to --center/--span)")
    p.add_argument("--points", type=int, default=500, help="Bins per sweep (default: 500)")
    p.add_argument("--rbw", type=float, default=10e3, help="Resolution bandwidth in Hz, 200 to 850e3 (default: 10 kHz)")
    p.add_argument("--interval", type=float, default=0.0,
                   help="Minimum seconds between logged points (default: 0, back-to-back sweeps)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Total seconds to log for (default: 0, run until Ctrl+C)")
    p.add_argument("--out", default="tinysa_power_log.csv", help="Output CSV path (default: tinysa_power_log.csv)")
    p.add_argument("--trace-out", help="Also write every point of every sweep to this CSV")
    args = p.parse_args()

    if args.start is not None and args.stop is not None:
        start, stop = args.start, args.stop
    elif args.center is not None and args.span is not None:
        start, stop = args.center - args.span / 2, args.center + args.span / 2
    else:
        p.error("specify either --start and --stop, or --center and --span")

    try:
        ser = open_tinysa(args.port)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"Connected to {ser.port}")
    command(ser, "pause")
    command(ser, f"rbw {args.rbw / 1e3:g}")

    print(f"Logging peak power {start/1e6:.4f}-{stop/1e6:.4f} MHz to {args.out} (Ctrl+C to stop)...")

    trace_f = open(args.trace_out, "w", newline="") if args.trace_out else None
    trace_writer = csv.writer(trace_f) if trace_f else None
    if trace_writer:
        trace_writer.writerow(["timestamp_utc", "sweep", "freq_hz", "dbm"])

    start_time = time.monotonic()
    sweep = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "elapsed_s", "peak_freq_hz", "peak_dbm", "peak_mw", "peak_uw"])
        f.flush()

        try:
            empty = 0
            while True:
                loop_start = time.monotonic()

                try:
                    pts = acquire(ser, start, stop, args.points)
                except RuntimeError as e:
                    empty += 1
                    if empty >= 5:
                        sys.exit(f"5 bad sweeps in a row, last error: {e}")
                    print(f"Bad sweep ({e}), retrying")
                    time.sleep(0.5)
                    continue
                empty = 0

                peak_freq, peak_dbm = max(pts, key=lambda t: t[1])
                peak_mw = 10 ** (peak_dbm / 10)
                peak_uw = peak_mw * 1000

                now = datetime.now(timezone.utc).isoformat()
                elapsed = time.monotonic() - start_time
                writer.writerow([now, f"{elapsed:.3f}", f"{peak_freq:.1f}", f"{peak_dbm:.3f}",
                                 f"{peak_mw:.6g}", f"{peak_uw:.6g}"])
                f.flush()
                if trace_f and trace_writer:
                    trace_writer.writerows([now, sweep, f"{fr:.1f}", f"{d:.3f}"] for fr, d in pts)
                    trace_f.flush()
                sweep += 1

                print(f"{now}  {peak_freq/1e6:.4f} MHz  {peak_dbm:7.3f} dBm  {peak_mw * 1e9:9.3f} pW")

                if args.duration and elapsed >= args.duration:
                    break

                remaining = args.interval - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\nStopped by user.")

    if trace_f:
        trace_f.close()
    command(ser, "resume")
    ser.close()
    print(f"Done. Data written to {args.out}")


if __name__ == "__main__":
    main()
