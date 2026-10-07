#!/usr/bin/env python3
"""
Compare set-single-bin.py output with one tinySA sweep, plotted vs R (span = 12 units).

Usage:
    1. python set-single-bin.py
    2. python compare_single_bin.py
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from log_tinysa_power import acquire, command, open_tinysa

HERE = Path(__file__).resolve().parent


def read_generated(script):
    text = Path(script).read_text()
    found = {}
    for key in ("target_freq_mhz", "target_power_dbm"):
        m = re.search(rf"^\s*{key}\s*=\s*([-+0-9.eE]+)", text, re.M)
        if not m:
            sys.exit(f"Could not find {key} in {script}")
        found[key] = float(m.group(1))
    return found["target_freq_mhz"] * 1e6, found["target_power_dbm"]


def measure(port, start_hz, stop_hz, bins, rbw_hz):
    ser = open_tinysa(port)
    try:
        command(ser, "pause")
        command(ser, f"rbw {rbw_hz / 1e3:g}")
        pts = acquire(ser, start_hz, stop_hz, bins)
        freqs = np.array([p[0] for p in pts])
        mw = 10 ** (np.array([p[1] for p in pts]) / 10)
        print(f"  peak {10 * np.log10(mw.max()):.1f} dBm")
        return freqs, mw
    finally:
        try:
            command(ser, "resume")
        except Exception:
            pass
        ser.close()


def _cross(freqs, mw, i_hi, i_lo, level):
    y_hi, y_lo = mw[i_hi], mw[i_lo]
    if y_hi == y_lo:
        return float(freqs[i_hi])
    t = (level - y_lo) / (y_hi - y_lo)
    return float(freqs[i_lo] + t * (freqs[i_hi] - freqs[i_lo]))


def width_at_level(freqs, mw, level):
    """Width in Hz between interpolated crossings of `level` around the peak."""
    i = int(np.argmax(mw))
    lo = hi = i
    while lo > 0 and mw[lo - 1] >= level:
        lo -= 1
    while hi < len(mw) - 1 and mw[hi + 1] >= level:
        hi += 1
    f_lo = float(freqs[0]) if lo == 0 else _cross(freqs, mw, lo, lo - 1, level)
    f_hi = float(freqs[-1]) if hi == len(mw) - 1 else _cross(freqs, mw, hi, hi + 1, level)
    return f_hi - f_lo, f_lo, f_hi


def analyse(freqs, mw, f_gen, gen_dbm):
    df = float(np.median(np.diff(freqs)))
    w_err = df * np.sqrt(2)  # ±df/2 per edge, added in quadrature

    peak_i = int(np.argmax(mw))
    peak_dbm = 10 * np.log10(mw[peak_i])
    noise_mw = float(np.median(mw))

    sig_w, sig_lo, sig_hi = width_at_level(freqs, mw, mw[peak_i] / 2)
    tot_w, tot_lo, tot_hi = width_at_level(freqs, mw, noise_mw * 10 ** (3 / 10))
    skirt_w = max(tot_w - sig_w, 0.0)
    skirt_err = np.hypot(w_err, w_err)

    return {
        "peak_freq_hz": freqs[peak_i],
        "peak_dbm": peak_dbm,
        "offset_khz": (freqs[peak_i] - f_gen) / 1e3,
        "loss_db": gen_dbm - peak_dbm,
        "noise_dbm": 10 * np.log10(noise_mw),
        "df_hz": df,
        "sig_w_hz": sig_w, "sig_w_err_hz": w_err, "sig_lo_hz": sig_lo, "sig_hi_hz": sig_hi,
        "tot_w_hz": tot_w, "tot_w_err_hz": w_err, "tot_lo_hz": tot_lo, "tot_hi_hz": tot_hi,
        "skirt_w_hz": skirt_w, "skirt_w_err_hz": skirt_err,
    }


def _khz(w_hz, err_hz):
    return f"{w_hz / 1e3:.2f} ± {err_hz / 1e3:.2f} kHz"


def plot(freqs, mw, f_gen, rbw_hz, stats, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc, per_r = (freqs[0] + freqs[-1]) / 2, (freqs[-1] - freqs[0]) / 12.0
    r = (freqs - fc) / per_r
    dbm = 10 * np.log10(mw)
    r_mark = lambda f: (f - fc) / per_r

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(r, dbm, color="#e08a1e", lw=1.2)
    ax.axvline(r_mark(f_gen), color="#3a7d5c", ls=":", lw=1.2, label="generated frequency")
    ax.axhline(stats["noise_dbm"], color="gray", ls="--", lw=1, label="median noise floor")
    for f_lo, f_hi in ((stats["sig_lo_hz"], stats["sig_hi_hz"]), (stats["tot_lo_hz"], stats["tot_hi_hz"])):
        ax.axvspan(r_mark(f_lo), r_mark(f_hi), color="black", alpha=0.06)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("R")
    ax.set_ylabel("measured power (dBm)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98, 0.97,
        f"-3 dB width: {_khz(stats['sig_w_hz'], stats['sig_w_err_hz'])}\n"
        f"total width (above noise+3 dB): {_khz(stats['tot_w_hz'], stats['tot_w_err_hz'])}\n"
        f"surrounding skirt: {_khz(stats['skirt_w_hz'], stats['skirt_w_err_hz'])}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax.set_title(f"Single sweep, {per_r / 1e3:.2f} kHz/R, RBW {rbw_hz / 1e3:g} kHz")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default=str(HERE / "set-single-bin.py"))
    p.add_argument("--freq", type=float, help="generated frequency in MHz")
    p.add_argument("--power", type=float, help="generated level in dBm")
    p.add_argument("--span", type=float, default=400.0, help="span in kHz")
    p.add_argument("--bins", type=int, default=500)
    p.add_argument("--rbw", type=float, default=1.0, help="RBW in kHz")
    p.add_argument("--port", help="serial port (default: auto-detect)")
    p.add_argument("--out", default=str(HERE / "compare_single_bin.png"))
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    f_gen, gen_dbm = read_generated(args.script)
    if args.freq is not None:
        f_gen = args.freq * 1e6
    if args.power is not None:
        gen_dbm = args.power
    print(f"Generated: {f_gen / 1e6:.4f} MHz at {gen_dbm:g} dBm")

    start, stop = f_gen - args.span * 1e3 / 2, f_gen + args.span * 1e3 / 2
    rbw_hz = args.rbw * 1e3
    try:
        freqs, mw = measure(args.port, start, stop, args.bins, rbw_hz)
    except RuntimeError as e:
        sys.exit(str(e))

    stats = analyse(freqs, mw, f_gen, gen_dbm)
    s = stats
    print(
        f"Peak {s['peak_freq_hz'] / 1e6:.4f} MHz ({s['offset_khz']:+.2f} kHz), {s['peak_dbm']:.1f} dBm, "
        f"-3 dB {_khz(s['sig_w_hz'], s['sig_w_err_hz'])}, "
        f"total {_khz(s['tot_w_hz'], s['tot_w_err_hz'])}, "
        f"skirt {_khz(s['skirt_w_hz'], s['skirt_w_err_hz'])}"
    )

    fc, per_r = (freqs[0] + freqs[-1]) / 2, (freqs[-1] - freqs[0]) / 12.0
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "R", "measured_dbm", "measured_mw"])
        for fr, m in zip(freqs, mw):
            w.writerow([f"{fr:.1f}", f"{(fr - fc) / per_r:.4f}", f"{10 * np.log10(m):.3f}", f"{m:.6g}"])
    print(f"Saved {csv_path}")

    try:
        plot(freqs, mw, f_gen, rbw_hz, stats, args.out, show=not args.no_show)
    except ImportError:
        sys.exit("matplotlib is needed for the plot: pip install matplotlib")


if __name__ == "__main__":
    main()
