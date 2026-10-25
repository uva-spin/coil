#!/usr/bin/env python3
"""Small combined RF-generator / tinySA logger with a safe toy-data mode.

Run with ``python3 coil_response_gui.py``. Hardware dependencies (pyvisa and
pyserial) are imported only when Hardware mode is selected. ``--demo-once``
creates one toy spectrum and CSV pair without opening a window or hardware.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


RBW_CHOICES_KHZ = (3, 10, 30, 100, 300, 600)


def dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return text or "coil_response"


class ToyBackend:
    """Deterministic synthetic resonator spectrum; it never opens hardware."""

    def __init__(self, config: dict):
        self.config = config
        self.sweep_number = 0

    def start(self):
        return

    def read_sweep(self):
        c = self.config
        n = c["points"]
        f0 = c["center_hz"]
        span = c["span_hz"]
        width = max(c["rbw_hz"] * 5.0, span / 24.0)
        peak = c["source_power_dbm"] - 40.0
        freqs, dbms = [], []
        for i in range(n):
            f = f0 - span / 2 + span * i / (n - 1)
            if c["mode"].startswith("Response"):
                # Illustrative coil transfer curve, not a physical calibration.
                x = 2.0 * (f - f0) / width
                response_db = -10.0 * math.log10(1.0 + x * x)
                ripple = 0.32 * math.sin(i * 0.17) + 0.14 * math.sin(i * 0.047)
                level = max(-120.0, peak + response_db + ripple)
            else:
                # A fixed CW tone appears as an RBW-shaped peak in a spectrum scan.
                source_f = c["source_freq_hz"]
                x = (f - source_f) / max(c["rbw_hz"] * 0.5, 1.0)
                response_db = -10.0 * math.log10(1.0 + x ** 4)
                level = max(-120.0, peak + response_db - 0.25 * abs(math.sin(i * 0.11)))
            freqs.append(f)
            dbms.append(level)
        self.sweep_number += 1
        return freqs, dbms

    def close(self):
        return


class HardwareBackend:
    """SCPI RF-generator setup plus the repo's tinySA serial sweep reader."""

    def __init__(self, config: dict):
        self.config = config
        self.rm = None
        self.generator = None
        self.ser = None

    def start(self):
        tiny_dir = str(Path(__file__).resolve().parent / "tinySA")
        if tiny_dir not in sys.path:
            sys.path.insert(0, tiny_dir)
        try:
            import pyvisa
            from log_tinysa_power import command, open_tinysa
        except ImportError as exc:
            raise RuntimeError("Hardware mode needs pyvisa and pyserial installed.") from exc

        c = self.config
        self._command = command
        self.rm = pyvisa.ResourceManager()
        self.generator = self.rm.open_resource(c["visa_address"], timeout=5000)
        # Keep the RF output off while configuring both instruments.
        self.generator.write("OUTPut:STATe OFF")
        self.generator.write(f"SOURce:FREQuency:CW {c['source_freq_hz']:.1f}")
        self.generator.write(f"SOURce:POWer:LEVel {c['source_power_dbm']:.3f}")
        actual_f = float(self.generator.query("SOURce:FREQuency:CW?"))
        actual_p = float(self.generator.query("SOURce:POWer:LEVel?"))
        c["source_freq_hz"] = actual_f
        c["source_power_dbm"] = actual_p

        self.ser = open_tinysa(c["serial_port"] or None)
        command(self.ser, "pause")
        command(self.ser, f"rbw {c['rbw_hz'] / 1e3:g}")
        # Enable the generator only after configuration and analyzer connection succeed.
        self.generator.write("OUTPut:STATe ON")

    def read_sweep(self):
        from log_tinysa_power import acquire
        c = self.config
        if c["mode"].startswith("Response"):
            frequencies = [c["center_hz"] - c["span_hz"] / 2
                           + c["span_hz"] * i / (c["points"] - 1)
                           for i in range(c["points"])]
            levels = []
            local_span = max(c["rbw_hz"], 1.0)
            for f_hz in frequencies:
                self.generator.write(f"SOURce:FREQuency:CW {f_hz:.1f}")
                time.sleep(c["settle_ms"] / 1000.0)
                pts = acquire(self.ser, f_hz - local_span / 2,
                              f_hz + local_span / 2, 3)
                levels.append(max(dbm for _freq, dbm in pts))
            return frequencies, levels
        pts = acquire(self.ser, c["center_hz"] - c["span_hz"] / 2,
                      c["center_hz"] + c["span_hz"] / 2, c["points"])
        return [p[0] for p in pts], [p[1] for p in pts]

    def close(self):
        if self.generator is not None:
            try:
                self.generator.write("OUTPut:STATe OFF")
            except Exception:
                pass
        if self.ser is not None:
            try:
                self._command(self.ser, "resume")
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
        if self.generator is not None:
            try:
                self.generator.close()
            except Exception:
                pass
        if self.rm is not None:
            try:
                self.rm.close()
            except Exception:
                pass


def create_run_files(outdir: Path, label: str):
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{safe_label(label)}_{timestamp_tag()}"
    run_n = 1
    stem = f"{base}_run-{run_n:03d}"
    while (outdir / f"{stem}_summary.csv").exists() or (outdir / f"{stem}_trace.csv").exists():
        run_n += 1
        stem = f"{base}_run-{run_n:03d}"
    summary = outdir / f"{stem}_summary.csv"
    trace = outdir / f"{stem}_trace.csv"
    return summary, trace


def write_sweep(summary_path: Path, trace_path: Path, config: dict,
                run_id: str, sweep_id: int, elapsed_s: float,
                freqs: list[float], dbms: list[float]):
    peak_i = max(range(len(dbms)), key=dbms.__getitem__)
    stamp = datetime.now(timezone.utc).isoformat()
    peak_dbm = dbms[peak_i]
    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        sweep_start = (config["center_hz"] - config["span_hz"] / 2
                       if config["mode"].startswith("Response") else config["source_freq_hz"])
        sweep_stop = (config["center_hz"] + config["span_hz"] / 2
                      if config["mode"].startswith("Response") else config["source_freq_hz"])
        w.writerow([run_id, config["backend"], config["mode"], sweep_id, stamp, f"{elapsed_s:.3f}",
                    f"{sweep_start:.1f}", f"{sweep_stop:.1f}", f"{config['source_power_dbm']:.3f}",
                    f"{config['center_hz']:.1f}", f"{config['span_hz']:.1f}",
                    config["points"], f"{config['rbw_hz']:.1f}",
                    f"{freqs[peak_i]:.1f}", f"{peak_dbm:.3f}", f"{dbm_to_mw(peak_dbm):.9g}"])
    with trace_path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerows([run_id, config["backend"], config["mode"], sweep_id, stamp, i, f"{freq:.1f}", f"{dbm:.3f}",
                     f"{dbm_to_mw(dbm):.9g}", f"{config['rbw_hz']:.1f}",
                     f"{freq if config['mode'].startswith('Response') else config['source_freq_hz']:.1f}",
                     f"{config['source_power_dbm']:.3f}"]
                    for i, (freq, dbm) in enumerate(zip(freqs, dbms)))


def make_csvs(summary: Path, trace: Path):
    with summary.open("w", newline="") as f:
        csv.writer(f).writerow([
            "run_id", "data_source", "mode", "sweep_id", "timestamp_utc", "elapsed_s",
            "source_sweep_start_hz", "source_sweep_stop_hz", "source_set_power_dbm",
            "analyzer_center_hz", "analyzer_span_hz", "points",
            "rbw_hz", "peak_freq_hz", "peak_dbm", "peak_mw"])
    with trace.open("w", newline="") as f:
        csv.writer(f).writerow([
            "run_id", "data_source", "mode", "sweep_id", "timestamp_utc", "bin_index", "freq_hz", "dbm",
            "mw", "rbw_hz", "source_freq_hz", "source_set_power_dbm"])


def demo_once(outdir: Path):
    c = dict(points=500, center_hz=213e6, span_hz=400e3, rbw_hz=3e3,
             source_freq_hz=213e6, source_power_dbm=-10.0,
             mode="Response: stepped source")
    label = "toy_coil"
    summary, trace = create_run_files(outdir, label)
    make_csvs(summary, trace)
    backend = ToyBackend(c)
    backend.start()
    t0 = time.monotonic()
    freqs, dbms = backend.read_sweep()
    c["backend"] = "Demo data"
    run_id = summary.stem.removesuffix("_summary")
    write_sweep(summary, trace, c, run_id, 1, time.monotonic() - t0, freqs, dbms)
    print(f"Toy sweep: {len(freqs)} bins; peak {max(dbms):.2f} dBm")
    print(f"Summary: {summary}")
    print(f"Trace:   {trace}")


class CoilResponseApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Coil Response Logger")
        root.geometry("1120x760")
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.closing = False
        self.latest_freqs = []
        self.latest_dbms = []
        self.display_unit = tk.StringVar(value="dBm")
        self.status = tk.StringVar(value="Ready — Demo mode is selected; no hardware will be opened.")
        self.mode = tk.StringVar(value="Demo data")
        self.measurement = tk.StringVar(value="Response: stepped source")
        self._build()
        preview = ToyBackend({
            "points": 500, "center_hz": 213e6, "span_hz": 400e3, "rbw_hz": 3e3,
            "source_freq_hz": 213e6, "source_power_dbm": -10.0,
            "mode": "Response: stepped source",
        })
        self.latest_freqs, self.latest_dbms = preview.read_sweep()
        self._refresh_readout()
        self.status.set("Toy preview loaded. Select Demo data and Start run to save CSVs; no hardware is connected.")
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        form = ttk.LabelFrame(outer, text="Run settings", padding=10)
        form.pack(fill="x")

        self.vars = {
            "label": tk.StringVar(value="coil_response"),
            "outdir": tk.StringVar(value=str(Path(__file__).resolve().parent / "coil_runs")),
            "visa": tk.StringVar(value="TCPIP0::192.168.1.10::inst0::INSTR"),
            "port": tk.StringVar(value=""),
            "source_mhz": tk.StringVar(value="213.000"),
            "source_dbm": tk.StringVar(value="-10.0"),
            "center_mhz": tk.StringVar(value="213.000"),
            "span_khz": tk.StringVar(value="400"),
            "points": tk.StringVar(value="500"),
            "rbw_khz": tk.StringVar(value="3"),
            "interval_s": tk.StringVar(value="60"),
            "settle_ms": tk.StringVar(value="50"),
        }
        rows = [
            [("Data source", "backend", "combo"), ("Run name", "label", "entry"),
             ("Output folder", "outdir", "folder")],
            [("Measurement", "measurement", "measurement"),
             ("Generator VISA", "visa", "entry"), ("tinySA serial port", "port", "entry")],
            [("Source freq (MHz)", "source_mhz", "entry"), ("Source set power (dBm)", "source_dbm", "entry"),
             ("Analyzer center (MHz)", "center_mhz", "entry")],
            [("Span (kHz)", "span_khz", "entry"), ("Points", "points", "entry"),
             ("RBW (kHz)", "rbw_khz", "rbw")],
            [("Interval (s)", "interval_s", "entry"), ("Settle (ms)", "settle_ms", "entry"),
             ("Display", "unit", "unit")],
        ]
        for ri, row in enumerate(rows):
            for ci, (label, key, kind) in enumerate(row):
                cell = ttk.Frame(form)
                cell.grid(row=ri, column=ci, sticky="ew", padx=4, pady=4)
                ttk.Label(cell, text=label).pack(anchor="w")
                if kind == "combo":
                    ttk.Combobox(cell, textvariable=self.mode, values=("Demo data", "Hardware"),
                                 state="readonly", width=19).pack(fill="x")
                elif kind == "measurement":
                    ttk.Combobox(cell, textvariable=self.measurement,
                                 values=("Response: stepped source", "Spectrum: fixed source"),
                                 state="readonly", width=25).pack(fill="x")
                elif kind == "rbw":
                    ttk.Combobox(cell, textvariable=self.vars["rbw_khz"],
                                 values=tuple(str(x) for x in RBW_CHOICES_KHZ),
                                 state="readonly", width=16).pack(fill="x")
                elif kind == "unit":
                    unit_box = ttk.Combobox(cell, textvariable=self.display_unit, values=("dBm", "mW"),
                                            state="readonly", width=15)
                    unit_box.pack(fill="x")
                    unit_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_readout())
                elif kind == "folder":
                    line = ttk.Frame(cell)
                    line.pack(fill="x")
                    ttk.Entry(line, textvariable=self.vars[key], width=28).pack(side="left", fill="x", expand=True)
                    ttk.Button(line, text="…", width=3, command=self._browse).pack(side="left", padx=(3, 0))
                else:
                    ttk.Entry(cell, textvariable=self.vars[key], width=21).pack(fill="x")
        for i in range(3):
            form.columnconfigure(i, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(10, 6))
        self.start_btn = ttk.Button(controls, text="Start run", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Stop after current sweep", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.readout = ttk.Label(controls, text="Peak: —", font=("TkDefaultFont", 12, "bold"))
        self.readout.pack(side="right")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(outer, background="#101820", highlightthickness=0, height=440)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_plot())
        self._draw_plot()

    def _browse(self):
        folder = filedialog.askdirectory(initialdir=self.vars["outdir"].get() or str(Path.cwd()))
        if folder:
            self.vars["outdir"].set(folder)

    def _read_config(self):
        try:
            c = {
                "label": self.vars["label"].get().strip() or "coil_response",
                "outdir": Path(self.vars["outdir"].get()).expanduser(),
                "visa_address": self.vars["visa"].get().strip(),
                "serial_port": self.vars["port"].get().strip(),
                "source_freq_hz": float(self.vars["source_mhz"].get()) * 1e6,
                "source_power_dbm": float(self.vars["source_dbm"].get()),
                "center_hz": float(self.vars["center_mhz"].get()) * 1e6,
                "span_hz": float(self.vars["span_khz"].get()) * 1e3,
                "points": int(self.vars["points"].get()),
                "rbw_hz": float(self.vars["rbw_khz"].get()) * 1e3,
                "interval_s": float(self.vars["interval_s"].get()),
                "settle_ms": float(self.vars["settle_ms"].get()),
                "mode": self.measurement.get(),
                "backend": self.mode.get(),
            }
            if c["points"] < 2 or c["points"] > 5000:
                raise ValueError("Points must be between 2 and 5000.")
            if c["span_hz"] <= 0 or c["rbw_hz"] <= 0 or c["interval_s"] < 0 or c["settle_ms"] < 0:
                raise ValueError("Span and RBW must be positive; interval cannot be negative.")
            c["outdir"].mkdir(parents=True, exist_ok=True)
            if c["backend"] == "Hardware" and not c["visa_address"]:
                raise ValueError("Enter the RF generator VISA resource address.")
            return c
        except (ValueError, OSError) as exc:
            raise ValueError(str(exc))

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.stop_event.clear()
        config["stop_event"] = self.stop_event
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set("Preparing run…")
        self.worker = threading.Thread(target=self._run, args=(config,), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status.set("Stopping after the current sweep…")
        self.stop_btn.configure(state="disabled")

    def _run(self, c):
        summary, trace = create_run_files(c["outdir"], c["label"])
        run_id = summary.stem.removesuffix("_summary")
        make_csvs(summary, trace)
        backend = ToyBackend(c) if c["backend"] == "Demo data" else HardwareBackend(c)
        try:
            backend.start()
            self.events.put(("status", f"Running {run_id}; saving {c['points']} points per sweep."))
            t0 = time.monotonic()
            n = 0
            while not self.stop_event.is_set():
                sweep_start = time.monotonic()
                freqs, dbms = backend.read_sweep()
                n += 1
                elapsed = time.monotonic() - t0
                write_sweep(summary, trace, c, run_id, n, elapsed, freqs, dbms)
                self.events.put(("sweep", freqs, dbms, n, summary, trace))
                if self.stop_event.is_set():
                    break
                wait_s = max(0.0, c["interval_s"] - (time.monotonic() - sweep_start))
                self.stop_event.wait(wait_s)
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            try:
                backend.close()
            except Exception:
                pass
            self.events.put(("done", summary, trace))

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "status":
                    self.status.set(event[1])
                elif event[0] == "sweep":
                    _, freqs, dbms, n, summary, trace = event
                    self.latest_freqs, self.latest_dbms = freqs, dbms
                    self._refresh_readout()
                    self.status.set(f"Saved sweep {n}: {summary.name} and {trace.name}")
                    self._draw_plot()
                elif event[0] == "error":
                    self.status.set(f"Error: {event[1]}")
                elif event[0] == "done":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    if self.status.get().startswith("Stopping"):
                        self.status.set(f"Stopped. CSVs saved: {event[1].name}, {event[2].name}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _refresh_readout(self):
        if not self.latest_dbms:
            return
        peak_i = max(range(len(self.latest_dbms)), key=self.latest_dbms.__getitem__)
        if self.display_unit.get() == "mW":
            self.readout.configure(text=f"Peak: {dbm_to_mw(self.latest_dbms[peak_i]):.6g} mW at {self.latest_freqs[peak_i]/1e6:.6f} MHz")
        else:
            self.readout.configure(text=f"Peak: {self.latest_dbms[peak_i]:.2f} dBm at {self.latest_freqs[peak_i]/1e6:.6f} MHz")
        self._draw_plot()

    def _draw_plot(self):
        canvas = self.canvas
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 160)
        left, right, top, bottom = 68, width - 20, 28, height - 45
        title = "Latest tinySA trace" if self.mode.get() == "Hardware" else "Latest toy-data trace"
        canvas.create_text(16, 12, anchor="nw", fill="#d9e2ec", text=title)
        if not self.latest_dbms:
            canvas.create_text(width / 2, height / 2, fill="#9fb3c8", text="Start a run to display a spectrum")
            return
        if self.display_unit.get() == "mW":
            values = [dbm_to_mw(x) for x in self.latest_dbms]
            ylabel = "Power (mW)"
            lo, hi = min(values), max(values)
            if hi == lo:
                hi = lo + 1e-12
            ticks = [lo + (hi - lo) * i / 4 for i in range(5)]
            fmt = lambda v: f"{v:.2g}"
        else:
            values = self.latest_dbms
            ylabel = "Power (dBm)"
            lo, hi = min(values), max(values)
            if hi - lo < 1:
                hi = lo + 1
            ticks = [lo + (hi - lo) * i / 4 for i in range(5)]
            fmt = lambda v: f"{v:.1f}"
        for i, v in enumerate(ticks):
            y = bottom - (v - lo) / (hi - lo) * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#283b4c")
            canvas.create_text(left - 8, y, anchor="e", fill="#9fb3c8", text=fmt(v))
        f0, f1 = self.latest_freqs[0] / 1e6, self.latest_freqs[-1] / 1e6
        points = []
        for i, value in enumerate(values):
            x = left + i / max(len(values) - 1, 1) * (right - left)
            y = bottom - (value - lo) / (hi - lo) * (bottom - top)
            points.extend((x, y))
        canvas.create_line(*points, fill="#45c2b1", width=2)
        canvas.create_line(left, bottom, right, bottom, fill="#9fb3c8")
        canvas.create_line(left, top, left, bottom, fill="#9fb3c8")
        canvas.create_text((left + right) / 2, height - 17, fill="#d9e2ec", text=f"Frequency (MHz): {f0:.4f} – {f1:.4f}")
        canvas.create_text(left, top - 3, anchor="sw", fill="#d9e2ec", text=ylabel)

    def _close(self):
        self.stop_event.set()
        self.closing = True
        if self.worker and self.worker.is_alive():
            self.status.set("Stopping safely; turning RF output off after the current sweep…")
            self.root.after(100, self._wait_for_close)
        else:
            self.root.destroy()

    def _wait_for_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(100, self._wait_for_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-once", action="store_true", help="write one toy sweep to CSV and exit")
    parser.add_argument("--outdir", type=Path, default=Path("coil_runs"), help="folder for --demo-once CSVs")
    args = parser.parse_args()
    if args.demo_once:
        demo_once(args.outdir)
        return
    root = tk.Tk()
    CoilResponseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
