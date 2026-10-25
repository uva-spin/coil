#!/usr/bin/env python3
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from log_tinysa_power import acquire, command, find_ports, open_tinysa


def power_unit(mw):
    """Pick a unit so the value reads between 1 and 1000. Returns (factor from mW, label)."""
    for factor, label in ((1e-3, "W"), (1, "mW"), (1e3, "uW"), (1e6, "nW"), (1e9, "pW")):
        if mw * factor >= 1:
            return factor, label
    return 1e12, "fW"


def fmt_power(mw):
    factor, label = power_unit(mw)
    return f"{mw * factor:.3g} {label}"


def distribution_metrics(freqs_hz, dbm, tone_hz):
    """Return per-sweep power-distribution metrics from a spectrum trace."""
    freqs = np.asarray(freqs_hz, dtype=float)
    dbm = np.asarray(dbm, dtype=float)
    if len(freqs) < 3 or len(freqs) != len(dbm):
        return {}

    mw = np.power(10.0, dbm / 10.0)
    edge_n = max(2, len(mw) // 10)
    floor_mw = float(np.median(np.r_[mw[:edge_n], mw[-edge_n:]]))
    signal_mw = np.maximum(mw - floor_mw, 0.0)
    if signal_mw.sum() <= 0:
        signal_mw = mw.copy()
    total_mw = float(signal_mw.sum())
    center_idx = int(np.argmin(np.abs(freqs - tone_hz)))
    offsets_hz = freqs - tone_hz
    adjacent = [i for i in (center_idx - 1, center_idx + 1) if 0 <= i < len(mw)]
    neighborhood = slice(max(0, center_idx - 1), min(len(mw), center_idx + 2))
    order = np.argsort(np.abs(offsets_hz))
    cumulative = np.cumsum(signal_mw[order]) / total_mw
    radius_i = min(int(np.searchsorted(cumulative, 0.90)), len(order) - 1)
    above_3db = signal_mw >= float(signal_mw.max()) * 10 ** (-3.0 / 10.0)
    bin_spacing_hz = float(np.mean(np.diff(freqs)))
    return {
        "center_bin_dbm": float(dbm[center_idx]),
        "center_bin_share_pct": 100.0 * float(signal_mw[center_idx]) / total_mw,
        "adjacent_leakage_pct": 100.0 * float(signal_mw[adjacent].sum()) / total_mw,
        "center_pm1_share_pct": 100.0 * float(signal_mw[neighborhood].sum()) / total_mw,
        "tone_to_bin_offset_hz": abs(float(offsets_hz[center_idx])),
        "width_3db_hz": float(freqs[above_3db].max() - freqs[above_3db].min()),
        "width_90_hz": 2.0 * abs(float(offsets_hz[order[radius_i]])),
        "bin_spacing_hz": bin_spacing_hz,
    }


def update_running_stats(state, value):
    """Update [count, mean, M2] using Welford's stable online algorithm."""
    count, mean, m2 = state
    count += 1
    delta = value - mean
    mean += delta / count
    m2 += delta * (value - mean)
    state[:] = [count, mean, m2]


def running_sample_std(state):
    return float(np.sqrt(state[2] / (state[0] - 1))) if state[0] > 1 else None


class LoggerThread(QThread):
    sweep = pyqtSignal(list, list, float, float, float)  # freqs_hz, dbms, elapsed_s, peak_hz, peak_dbm
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop = False
        self._failed = False

    def stop(self):
        self._stop = True

    def run(self):
        c = self.cfg
        ser = generator = rm = trace_f = None
        command_fn = command
        run_stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", c["run_name"].strip()).strip("_") or "coil_response"
        stem_base = f"{label}_{run_stamp}"
        outdir = Path(c["outdir"]).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        run_num = 1
        stem = f"{stem_base}_run-{run_num:03d}"
        while (outdir / f"{stem}_summary.csv").exists() or (outdir / f"{stem}_trace.csv").exists():
            run_num += 1
            stem = f"{stem_base}_run-{run_num:03d}"
        summary_path, trace_path = outdir / f"{stem}_summary.csv", outdir / f"{stem}_trace.csv"
        run_id = stem
        self.status.emit("Preparing run files...")
        try:
            if c["demo"]:
                self.status.emit("Demo mode — no instruments will be opened.")
            else:
                import pyvisa
                is_sma = c["generator_profile"] == "R&S SMA (internal FM)"
                if c["fm_enabled"] and not is_sma:
                    raise RuntimeError("FM requires the R&S SMA (internal FM) generator profile.")
                if c["fm_enabled"] and c["measurement_mode"].startswith("Coil response"):
                    raise RuntimeError("Disable FM for stepped-source coil response; FM is used with a fixed carrier spectrum.")
                self.status.emit("Connecting to RF generator and tinySA...")
                rm = pyvisa.ResourceManager()
                generator = rm.open_resource(c["visa_address"], timeout=5000)
                freq_cmd = "SOURce1:FREQuency:CW" if is_sma else "SOURce:FREQuency:CW"
                power_cmd = ("SOURce1:POWer:LEVel:IMMediate:AMPLitude" if is_sma
                             else "SOURce:POWer:LEVel")
                output_cmd = "OUTPut1:STATe" if is_sma else "OUTPut:STATe"
                generator.write(f"{output_cmd} OFF")
                generator.write(f"{freq_cmd} {c['source_freq_hz']:.1f}")
                generator.write(f"{power_cmd} {c['source_power_dbm']:.3f}")
                c["source_freq_hz"] = float(generator.query(f"{freq_cmd}?"))
                c["source_power_dbm"] = float(generator.query(f"{power_cmd}?"))
                if is_sma:
                    if c["fm_enabled"]:
                        waveform_scpi = {
                            "SINE": "SINE", "SQUARE": "SQUare", "PULSE": "PULSe",
                            "TRIANGLE": "TRIangle", "TRAPEZE": "TRAPeze",
                        }[c["fm_waveform"]]
                        generator.write(f"SOURce1:LFOutput1:SHAPe {waveform_scpi}")
                        generator.write(f"SOURce1:LFOutput1:FREQuency {c['fm_rate_hz']:.6g}")
                        generator.write("SOURce1:FM1:SOURce LF1")
                        generator.write(f"SOURce1:FM1:DEViation {c['fm_span_hz'] / 2:.6g}")
                        generator.write("SOURce1:FM:MODe HBANdwidth")
                        generator.write("SOURce1:FM1:STATe 1")
                        generator.write("SOURce1:LFOutput1:STATe 1")
                    else:
                        generator.write("SOURce1:FM1:STATe 0")
                        generator.write("SOURce1:LFOutput1:STATe 0")
                ser = open_tinysa(c["port"] or None)
                command_fn(ser, "pause")
                command_fn(ser, f"rbw {c['rbw_hz'] / 1e3:g}")
                generator.write(f"{output_cmd} ON")

            with summary_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["run_id", "data_source", "measurement_mode", "sweep_id", "timestamp_utc",
                            "elapsed_s", "n_sweeps_requested", "sweep_time_target_s", "sweep_elapsed_s",
                            "generator_profile", "fm_enabled", "fm_waveform", "fm_rate_hz", "fm_span_hz",
                            "source_start_hz", "source_stop_hz", "source_set_power_dbm",
                            "center_hz", "span_hz", "points", "rbw_hz", "bin_spacing_hz",
                            "peak_freq_hz", "peak_dbm", "peak_mw", "center_bin_dbm",
                            "center_bin_share_pct", "adjacent_leakage_pct", "center_pm1_share_pct",
                            "tone_to_bin_offset_hz", "width_3db_hz", "width_90_hz",
                            "sigma_center_bin_db", "sigma_adjacent_leakage_pct_points"])
            trace_f = trace_path.open("w", newline="")
            trace_w = csv.writer(trace_f)
            trace_w.writerow(["run_id", "data_source", "measurement_mode", "sweep_id", "timestamp_utc",
                              "sweep_elapsed_s", "bin_index", "freq_hz", "dbm", "mw", "rbw_hz",
                              "bin_spacing_hz", "source_freq_hz",
                              "source_set_power_dbm", "generator_profile", "fm_enabled", "fm_waveform",
                              "fm_rate_hz", "fm_span_hz"])

            start_time = time.monotonic()
            n = 0
            bad = 0
            center_bin_stats = [0, 0.0, 0.0]
            adjacent_leakage_stats = [0, 0.0, 0.0]
            self.status.emit(f"Run {run_id} — logging")
            with summary_path.open("a", newline="") as f:
                w = csv.writer(f)
                while not self._stop:
                    loop_start = time.monotonic()
                    try:
                        if c["demo"]:
                            freqs, dbms = self._toy_sweep(c)
                        elif c["measurement_mode"] == "Coil response (stepped source)":
                            freqs, dbms = self._step_response(c, generator, ser, command_fn)
                        else:
                            pts = acquire(ser, c["start_hz"], c["stop_hz"], c["points"])
                            freqs, dbms = [p[0] for p in pts], [p[1] for p in pts]
                    except RuntimeError as e:
                        bad += 1
                        if bad >= 5:
                            raise RuntimeError(f"5 failed sweeps in a row, last error: {e}")
                        self.status.emit(f"Bad sweep ({e}), retrying...")
                        time.sleep(0.5)
                        continue
                    bad = 0
                    if not freqs or not dbms:
                        break
                    sweep_elapsed = time.monotonic() - loop_start
                    i = max(range(len(dbms)), key=dbms.__getitem__)
                    peak_hz, peak_dbm = freqs[i], dbms[i]
                    peak_mw = 10 ** (peak_dbm / 10)
                    analysis_metrics = (distribution_metrics(freqs, dbms, c["source_freq_hz"])
                                        if c["measurement_mode"] in
                                        ("Single-tone calibration", "Modulated frequency profile") else {})
                    if c["measurement_mode"] == "Modulated frequency profile" and analysis_metrics:
                        analysis_metrics = {key: analysis_metrics[key] for key in
                                           ("bin_spacing_hz", "width_3db_hz", "width_90_hz")}
                    sigma_center_db = ""
                    sigma_adjacent_pct_points = ""
                    if c["measurement_mode"] == "Single-tone calibration" and analysis_metrics:
                        update_running_stats(center_bin_stats, analysis_metrics["center_bin_dbm"])
                        update_running_stats(adjacent_leakage_stats, analysis_metrics["adjacent_leakage_pct"])
                        center_sigma = running_sample_std(center_bin_stats)
                        leakage_sigma = running_sample_std(adjacent_leakage_stats)
                        if center_sigma is not None:
                            sigma_center_db = f"{center_sigma:.6g}"
                            sigma_adjacent_pct_points = f"{leakage_sigma:.6g}"
                    now = datetime.now(timezone.utc).isoformat()
                    elapsed = time.monotonic() - start_time
                    source_start = c["start_hz"] if c["measurement_mode"].startswith("Coil response") else c["source_freq_hz"]
                    source_stop = c["stop_hz"] if c["measurement_mode"].startswith("Coil response") else c["source_freq_hz"]
                    data_source = "toy" if c["demo"] else "hardware"
                    w.writerow([run_id, data_source, c["measurement_mode"], n, now, f"{elapsed:.3f}",
                                c["n_sweeps"], f"{c['sweep_time']:.3f}", f"{sweep_elapsed:.3f}",
                                c["generator_profile"], c["fm_enabled"], c["fm_waveform"],
                                f"{c['fm_rate_hz']:.6g}", f"{c['fm_span_hz']:.3f}",
                                f"{source_start:.1f}", f"{source_stop:.1f}", f"{c['source_power_dbm']:.3f}",
                                f"{c['center_hz']:.1f}", f"{c['span_hz']:.1f}", c["points"],
                                f"{c['rbw_hz']:.1f}",
                                f"{analysis_metrics.get('bin_spacing_hz', '')}",
                                f"{peak_hz:.1f}", f"{peak_dbm:.3f}", f"{peak_mw:.9g}",
                                f"{analysis_metrics.get('center_bin_dbm', '')}",
                                f"{analysis_metrics.get('center_bin_share_pct', '')}",
                                f"{analysis_metrics.get('adjacent_leakage_pct', '')}",
                                f"{analysis_metrics.get('center_pm1_share_pct', '')}",
                                f"{analysis_metrics.get('tone_to_bin_offset_hz', '')}",
                                f"{analysis_metrics.get('width_3db_hz', '')}",
                                f"{analysis_metrics.get('width_90_hz', '')}",
                                sigma_center_db, sigma_adjacent_pct_points])
                    f.flush()
                    trace_w.writerows([run_id, data_source, c["measurement_mode"], n, now,
                                       f"{sweep_elapsed:.3f}", j,
                                       f"{fr:.1f}", f"{d:.3f}", f"{10 ** (d / 10):.9g}",
                                       f"{c['rbw_hz']:.1f}",
                                       f"{analysis_metrics.get('bin_spacing_hz', '')}",
                                       f"{fr if c['measurement_mode'].startswith('Coil response') else c['source_freq_hz']:.1f}",
                                       f"{c['source_power_dbm']:.3f}", c["generator_profile"],
                                       c["fm_enabled"], c["fm_waveform"], f"{c['fm_rate_hz']:.6g}",
                                       f"{c['fm_span_hz']:.3f}"]
                                      for j, (fr, d) in enumerate(zip(freqs, dbms)))
                    trace_f.flush()
                    n += 1
                    requested = c["n_sweeps"]
                    self.status.emit(f"Run {run_id} — sweep {n}"
                                     + (f" / {requested}" if requested else ""))
                    self.sweep.emit(freqs, dbms, elapsed, peak_hz, peak_dbm)
                    if requested and n >= requested:
                        break
                    if c["duration"] and elapsed >= c["duration"]:
                        break
                    remaining = c["sweep_time"] - (time.monotonic() - loop_start)
                    while remaining > 0 and not self._stop:
                        pause_ms = min(100, max(1, int(remaining * 1000)))
                        self.msleep(pause_ms)
                        remaining -= pause_ms / 1000.0
        except Exception as e:
            self._failed = True
            self.failed.emit(str(e))
        finally:
            if trace_f:
                trace_f.close()
            if generator is not None:
                try:
                    if c["generator_profile"] == "R&S SMA (internal FM)":
                        generator.write("SOURce1:FM1:STATe 0")
                        generator.write("SOURce1:LFOutput1:STATe 0")
                        generator.write("OUTPut1:STATe OFF")
                    else:
                        generator.write("OUTPut:STATe OFF")
                except Exception:
                    pass
            if ser is not None:
                try:
                    command_fn(ser, "resume")
                except Exception:
                    pass
                ser.close()
            if generator is not None:
                try:
                    generator.close()
                except Exception:
                    pass
            if rm is not None:
                rm.close()
            if not self._failed:
                result = "Stopped" if self._stop else f"Completed {n} sweeps"
                self.status.emit(f"{result}. Saved {summary_path.name} and {trace_path.name}")

    @staticmethod
    def _toy_sweep(c):
        n = c["points"]
        center, span = c["center_hz"], c["span_hz"]
        width = max(c["rbw_hz"] * 5.0, span / 24.0)
        # Keep the demo's integrated receive power equal to the configured source power.
        # The normalized FM weights divide that total across the displayed bins.
        peak = c["source_power_dbm"]
        freqs, dbms = [], []
        fm_weights = None
        if c["fm_enabled"] and c["measurement_mode"] == "Modulated frequency profile":
            deviation = c["fm_span_hz"] / 2
            x = (np.linspace(center - span / 2, center + span / 2, n) - c["source_freq_hz"]) / max(deviation, 1.0)
            inside = np.abs(x) <= 1.0
            fm_weights = np.zeros(n, dtype=float)
            if c["fm_waveform"] == "TRIANGLE":
                fm_weights[inside] = 1.0
            elif c["fm_waveform"] == "SINE":
                fm_weights[inside] = 1.0 / np.sqrt(np.maximum(1.0 - x[inside] ** 2, 0.002))
            elif c["fm_waveform"] == "SQUARE":
                width = max(c["rbw_hz"] / max(deviation, 1.0), 0.005)
                fm_weights = np.exp(-0.5 * ((np.abs(x) - 1.0) / width) ** 2)
            elif c["fm_waveform"] == "PULSE":
                fm_weights[inside] = 0.35 + 0.65 * np.cos(np.pi * x[inside]) ** 2
            else:  # trapezoid: flat center with rounded, higher-density edges
                fm_weights[inside] = 1.0 + 0.35 * np.abs(x[inside]) ** 4
            if fm_weights.sum() > 0:
                # Each analyzer point reports power integrated over one RBW. Since
                # neighboring points overlap, scale so sum(power) * step / RBW
                # recovers the configured source power rather than under-counting it.
                bin_step_hz = span / max(n - 1, 1)
                fm_weights *= (c["rbw_hz"] / max(bin_step_hz, 1.0)) / fm_weights.sum()
        for i in range(n):
            freq = center - span / 2 + span * i / (n - 1)
            if c["measurement_mode"].startswith("Coil response"):
                x = 2.0 * (freq - center) / width
                level = peak - 10.0 * np.log10(1.0 + x * x) + 0.3 * np.sin(i * 0.17)
            elif fm_weights is not None:
                level = 10.0 * np.log10(max(10 ** (peak / 10) * fm_weights[i], 1e-30))
            else:
                x = (freq - c["source_freq_hz"]) / max(c["rbw_hz"] / 2, 1)
                level = peak - 10.0 * np.log10(1.0 + x ** 4)
            freqs.append(freq)
            dbms.append(float(max(-120, level)))
        return freqs, dbms

    @staticmethod
    def _step_response(c, generator, ser, command_fn):
        freqs, levels = [], []
        n = c["points"]
        freq_cmd = ("SOURce1:FREQuency:CW" if c["generator_profile"] == "R&S SMA (internal FM)"
                    else "SOURce:FREQuency:CW")
        for i in range(n):
            freq = c["start_hz"] + (c["stop_hz"] - c["start_hz"]) * i / (n - 1)
            generator.write(f"{freq_cmd} {freq:.1f}")
            time.sleep(c["settle_ms"] / 1000.0)
            points = acquire(ser, freq - c["rbw_hz"] / 2, freq + c["rbw_hz"] / 2, 3)
            freqs.append(freq)
            levels.append(max(p[1] for p in points))
        return freqs, levels


# NMR-AI 500-bin grid (configs/deuteron-acquisition.json) moved to the proton line:
# same 1.5287 kHz bin step, 500 bins (499 steps), centered on 213 MHz.
NMR_AI_BINS = 500
NMR_AI_STEP_KHZ = 1.5287
PROTON_CENTER_MHZ = 213.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("tinySA Power Logger")
        self.thread = None
        self.t_hist, self.p_hist = [], []
        self._run_measurement_name = ""
        self._single_tone_center_stats = [0, 0.0, 0.0]
        self._single_tone_leakage_stats = [0, 0.0, 0.0]
        self._repeatability_text = "Repeatability σ requires at least two single-tone sweeps."

        # ---- settings form ----
        self.port = QComboBox()
        self.port.setEditable(True)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_ports)
        port_row = QHBoxLayout()
        port_row.addWidget(self.port, 1)
        port_row.addWidget(refresh)

        self.data_source = QComboBox()
        self.data_source.addItems(["Demo data", "Hardware"])
        self.data_source.setToolTip(
            "Demo data are synthetic. The simulated received total power equals the source setting; "
            "no coil, cable, or coupling loss is included.")
        self.measurement_mode = QComboBox()
        self.measurement_mode.addItems(["CW / coil response", "Modulated frequency profile"])
        self.cw_measurement = QComboBox()
        self.cw_measurement.addItems(["Single-tone leakage check", "Coil response sweep"])
        self.cw_measurement.setCurrentIndex(1)
        self.generator_profile = QComboBox()
        self.generator_profile.addItems(["Generic CW (basic SCPI)", "R&S SMA (internal FM)"])
        self.visa = QLineEdit()
        self.visa.setPlaceholderText("USB0::...::INSTR")
        self.source_freq = self._spin(0.1, 6000, 213.0, 4, " MHz")
        self.source_power = self._spin(-120, 30, -20.0, 1, " dBm")
        self.fm_waveform = QComboBox()
        self.fm_waveform.addItems(["SINE", "SQUARE", "PULSE", "TRIANGLE", "TRAPEZE"])
        self.fm_waveform.setCurrentText("TRIANGLE")
        self.fm_rate = self._spin(0.001, 100000, 1.0, 3, " Hz")
        self.fm_span = self._spin(0.001, 1000000, 400.0, 3, " kHz")
        self.center = self._spin(0.1, 6000, 213.0, 4, " MHz")
        self.span = self._spin(1, 6e6, 400.0, 1, " kHz")
        self.rbw = QComboBox()
        self.rbw.addItems(["0.2", "0.3", "0.6", "1", "3", "10", "30", "100", "300", "850"])
        self.rbw.setCurrentText("10")
        self.points = QSpinBox()
        self.points.setRange(11, 1000)
        self.points.setValue(500)
        self.points.setToolTip(
            "Single-tone and modulation modes use an odd point count so the carrier frequency is sampled in a bin. "
            "Coil-response sweeps can use any count.")
        self.n_sweeps = QSpinBox()
        self.n_sweeps.setRange(0, 1000000)
        self.n_sweeps.setValue(0)
        self.n_sweeps.setToolTip("0 runs continuously until Stop or the total duration is reached.")
        self.sweep_time = self._spin(0, 3600, 60.0, 1, " s")
        self.sweep_time.setToolTip(
            "Target minimum time from the start of one full frequency sweep to the next. "
            "If acquisition takes longer, the actual sweep time is logged.")
        self.duration = self._spin(0, 86400, 0.0, 0, " s (0 = until Stop)")
        self.settle = self._spin(0, 10000, 20, 0, " ms")
        self.measurement_mode.currentIndexChanged.connect(self.update_mode_controls)
        self.cw_measurement.currentIndexChanged.connect(self.update_mode_controls)
        self.generator_profile.currentIndexChanged.connect(self.update_mode_controls)
        self.points.valueChanged.connect(self.update_mode_controls)
        self.source_freq.valueChanged.connect(self.sync_single_tone_center)
        self.update_mode_controls()

        preset = QPushButton("NMR-AI grid (proton, 500 bins)")
        preset.clicked.connect(self.apply_nmr_ai_preset)

        self.run_name = QLineEdit("coil_response")
        self.out = QLineEdit(str(Path.home() / "tinySA_runs"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out, 1)
        out_row.addWidget(browse)

        form = QFormLayout()
        form.addRow("Data source", self.data_source)
        form.addRow("Measurement", self.measurement_mode)
        form.addRow("CW measurement", self.cw_measurement)
        form.addRow("Generator profile", self.generator_profile)
        form.addRow("Port", port_row)
        form.addRow("Generator VISA", self.visa)
        form.addRow("Source frequency", self.source_freq)
        form.addRow("Source power", self.source_power)
        form.addRow("FM waveform", self.fm_waveform)
        form.addRow("FM rate", self.fm_rate)
        form.addRow("FM span", self.fm_span)
        form.addRow("Center", self.center)
        form.addRow("Span", self.span)
        form.addRow("RBW (kHz)", self.rbw)
        form.addRow("Bins / points", self.points)
        form.addRow("Source settle", self.settle)
        form.addRow("N sweeps (0 = continuous)", self.n_sweeps)
        form.addRow("Sweep time", self.sweep_time)
        form.addRow("Duration", self.duration)
        form.addRow(preset)
        form.addRow("Run name", self.run_name)
        form.addRow("Output folder", out_row)
        box = QGroupBox("Settings")
        box.setLayout(form)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)

        self.readout = QLabel("Peak: -")
        self.readout.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.state = QLabel("Idle")

        left = QVBoxLayout()
        left.addWidget(box)
        left.addLayout(btns)
        left.addWidget(self.readout)
        left.addWidget(self.state)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(left_w)
        settings_scroll.setMinimumWidth(390)
        settings_scroll.setMaximumWidth(440)

        # ---- plots ----
        self.trace_plot = pg.PlotWidget(title="Latest trace")
        self.trace_plot.setLabel("bottom", "Frequency", "MHz")
        self.trace_plot.setLabel("left", "Power", "dBm")
        self.trace_plot.showGrid(x=True, y=True)
        self.trace_curve = self.trace_plot.plot(pen=pg.mkPen("c", width=2))

        self.hist_plot = pg.PlotWidget(title="Peak power vs time")
        self.hist_plot.setLabel("bottom", "Time", "s")
        self.hist_plot.setLabel("left", "Peak", "dBm")
        self.hist_plot.showGrid(x=True, y=True)
        self.hist_curve = self.hist_plot.plot(pen=pg.mkPen("y", width=2), symbol="o", symbolSize=4)

        plots = QVBoxLayout()
        plots.addWidget(self.trace_plot, 3)
        plots.addWidget(self.hist_plot, 2)
        trace_tab = QWidget()
        trace_tab.setLayout(plots)

        # ---- power-in-bins tab ----
        self.bin_mode = QComboBox()
        self.bin_mode.addItems(["Latest sweep", "Running average"])
        self.bin_mode.currentIndexChanged.connect(self.redraw_bins)
        self.bin_units = QComboBox()
        self.bin_units.addItems(["power per bin (auto unit)", "mW per bin", "% of total power", "dBm per bin"])
        self.bin_units.currentIndexChanged.connect(self.redraw_bins)
        self.bin_x = QComboBox()
        self.bin_x.addItems(["MHz", "bin index j"])
        self.bin_x.currentIndexChanged.connect(self.redraw_bins)
        load = QPushButton("Load target profile...")
        load.clicked.connect(self.load_target)
        clear = QPushButton("Clear target")
        clear.clicked.connect(self.clear_target)
        reset = QPushButton("Reset average")
        reset.clicked.connect(self.reset_bins)
        self.bin_info = QLabel("No data yet")
        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("Show"))
        bin_row.addWidget(self.bin_mode)
        bin_row.addWidget(QLabel("Units"))
        bin_row.addWidget(self.bin_units)
        bin_row.addWidget(QLabel("X axis"))
        bin_row.addWidget(self.bin_x)
        bin_row.addWidget(load)
        bin_row.addWidget(clear)
        bin_row.addWidget(reset)
        bin_row.addStretch(1)

        self.bin_plot = pg.PlotWidget(title="Power per frequency bin")
        self.bin_plot.setLabel("bottom", "Frequency", "MHz")
        self.bin_plot.showGrid(x=True, y=True)
        self.bin_bars = pg.BarGraphItem(x=[0], height=[0], width=1, brush="c")
        self.bin_plot.addItem(self.bin_bars)
        self.bin_plot.addLegend()
        self.target_curve = self.bin_plot.plot(
            pen=pg.mkPen("r", width=2), stepMode=False, name="target profile")
        self.target_curve.setVisible(False)
        self.target_f = None  # MHz, from the loaded profile file
        self.target_u = None

        bins_lay = QVBoxLayout()
        bins_lay.addLayout(bin_row)
        bins_lay.addWidget(self.bin_plot, 1)
        bins_lay.addWidget(self.bin_info)
        bins_tab = QWidget()
        bins_tab.setLayout(bins_lay)
        self.bin_tab = bins_tab
        self.bin_freqs = None
        self.bin_latest_mw = None
        self.bin_sum_mw = None
        self.bin_count = 0

        # ---- target U(R) tab: the profile on the R axis (grid span maps linearly to R = -6..6) ----
        self.ur_plot = pg.PlotWidget(title="Power profile U(R)")
        self.ur_plot.setLabel("bottom", "R")
        self.ur_plot.setLabel("left", "U(R)")
        self.ur_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ur_plot.setXRange(-6, 6)
        self.ur_plot.addLegend()
        green = pg.mkColor("#3a7d5c")
        fill = pg.mkColor("#3a7d5c")
        fill.setAlpha(60)
        self.ur_target = self.ur_plot.plot(pen=pg.mkPen(green, width=2), brush=fill, fillLevel=0,
                                           name="target U(R)")
        self.ur_meas = self.ur_plot.plot(pen=pg.mkPen("#e08a1e", width=1.5, style=Qt.PenStyle.DashLine),
                                         name="measured")
        self.ur_meas.setVisible(False)
        self.ur_info = QLabel("Press Start to see the measured profile, or load a target profile on the Power bins tab.")
        ur_lay = QVBoxLayout()
        ur_lay.addWidget(self.ur_plot, 1)
        ur_lay.addWidget(self.ur_info)
        ur_tab = QWidget()
        ur_tab.setLayout(ur_lay)

        # ---- single-tone bin leakage tab ----
        self.resolution_plot = pg.PlotWidget(title="Frequency-bin distribution analysis")
        self.resolution_plot.setLabel("bottom", "Offset from source frequency", "kHz")
        self.resolution_plot.setLabel("left", "Relative power", "dBc")
        self.resolution_plot.showGrid(x=True, y=True)
        self.resolution_curve = self.resolution_plot.plot(pen=pg.mkPen("m", width=2),
                                                          symbol="o", symbolSize=3)
        self.resolution_info = QLabel("Select a single-tone leakage check or modulated profile to analyze bin spread.")
        resolution_lay = QVBoxLayout()
        resolution_lay.addWidget(self.resolution_plot, 1)
        resolution_lay.addWidget(self.resolution_info)
        resolution_tab = QWidget()
        resolution_tab.setLayout(resolution_lay)
        self.resolution_tab = resolution_tab
        self._run_source_freq_hz = self.source_freq.value() * 1e6

        self._bin_factor = 1e3
        self._rbw_hz = 10e3

        tabs = QTabWidget()
        tabs.addTab(trace_tab, "Trace")
        tabs.addTab(bins_tab, "Power bins")
        tabs.addTab(ur_tab, "U(R) profile")
        tabs.addTab(resolution_tab, "Distribution analysis")
        self.tabs = tabs

        root = QHBoxLayout()
        root.addWidget(settings_scroll)
        root.addWidget(tabs, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1200, 720)
        self.refresh_ports()

    @staticmethod
    def _spin(lo, hi, val, decimals, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def update_mode_controls(self, *_):
        mode = self.measurement_mode.currentIndex()
        stepped = mode == 0 and self.cw_measurement.currentIndex() == 1
        fixed_carrier = not stepped
        if mode == 1 and self.generator_profile.currentIndex() != 1:
            self.generator_profile.setCurrentIndex(1)
        if fixed_carrier:
            self.center.setValue(self.source_freq.value())
            if self.points.value() % 2 == 0:
                self.points.setValue(self.points.value() + 1 if self.points.value() < 1000
                                     else self.points.value() - 1)
        self.center.setEnabled(not fixed_carrier)
        self.source_freq.setEnabled(fixed_carrier)
        self.settle.setEnabled(stepped)
        modulation = mode == 1 and self.generator_profile.currentIndex() == 1
        self.generator_profile.setEnabled(mode != 1)
        self.cw_measurement.setEnabled(mode == 0)
        self.fm_waveform.setEnabled(modulation)
        self.fm_rate.setEnabled(modulation)
        self.fm_span.setEnabled(modulation)
        if hasattr(self, "resolution_curve"):
            self.redraw_bins()

    def sync_single_tone_center(self, value):
        if (self.measurement_mode.currentIndex() == 1
                or self.cw_measurement.currentIndex() == 0):
            self.center.setValue(value)

    def apply_nmr_ai_preset(self):
        if (self.measurement_mode.currentIndex() == 1
                or self.cw_measurement.currentIndex() == 0):
            self.source_freq.setValue(PROTON_CENTER_MHZ)
        else:
            self.center.setValue(PROTON_CENTER_MHZ)
        self.span.setValue((NMR_AI_BINS - 1) * NMR_AI_STEP_KHZ)
        self.points.setValue(NMR_AI_BINS)
        self.rbw.setCurrentText("3")

    def refresh_ports(self):
        self.port.clear()
        self.port.addItems(find_ports())

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if path:
            self.out.setText(path)

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def start(self):
        demo = self.data_source.currentIndex() == 0
        self._run_source_freq_hz = self.source_freq.value() * 1e6
        mode = self.measurement_mode.currentIndex()
        stepped = mode == 0 and self.cw_measurement.currentIndex() == 1
        fm_enabled = mode == 1
        if not demo and fm_enabled and self.generator_profile.currentIndex() != 1:
            self.state.setText("Modulated frequency profile requires the R&S SMA generator profile.")
            return
        if not demo and not self.visa.text().strip():
            self.state.setText("Enter the RF generator VISA address, or select Demo data.")
            return
        if not demo and not self.port.currentText().strip():
            self.refresh_ports()
        port = self.port.currentText().strip()  # empty means auto-detect when the logger starts
        center, span = self.center.value() * 1e6, self.span.value() * 1e3
        source_freq = self.source_freq.value() * 1e6
        rbw_hz = float(self.rbw.currentText()) * 1e3
        measurement_name = ("Modulated frequency profile" if fm_enabled else
                            "Coil response (stepped source)" if stepped else
                            "Single-tone calibration")
        cfg = {
            "demo": demo,
            "measurement_mode": measurement_name,
            "generator_profile": self.generator_profile.currentText(),
            "fm_enabled": fm_enabled,
            "fm_waveform": self.fm_waveform.currentText(),
            "fm_rate_hz": self.fm_rate.value(),
            "fm_span_hz": self.fm_span.value() * 1e3,
            "port": port,
            "visa_address": self.visa.text().strip(),
            "source_freq_hz": source_freq,
            "source_power_dbm": self.source_power.value(),
            "center_hz": center,
            "span_hz": span,
            "start_hz": center - span / 2,
            "stop_hz": center + span / 2,
            "rbw_hz": rbw_hz,
            "points": self.points.value(),
            "n_sweeps": self.n_sweeps.value(),
            "sweep_time": self.sweep_time.value(),
            "duration": self.duration.value(),
            "settle_ms": self.settle.value(),
            "outdir": self.out.text().strip() or str(Path.home() / "tinySA_runs"),
            "run_name": self.run_name.text().strip() or "coil_response",
        }
        self._run_measurement_name = measurement_name
        self._single_tone_center_stats = [0, 0.0, 0.0]
        self._single_tone_leakage_stats = [0, 0.0, 0.0]
        self._repeatability_text = "Repeatability σ requires at least two single-tone sweeps."
        self.t_hist, self.p_hist = [], []
        self.hist_curve.setData([], [])
        self.reset_bins()
        self._rbw_hz = rbw_hz
        self.tabs.setCurrentWidget(self.resolution_tab if not stepped else self.bin_tab)
        self.thread = LoggerThread(cfg)
        self.thread.sweep.connect(self.on_sweep)
        self.thread.status.connect(self.state.setText)
        self.thread.failed.connect(self.on_failed)
        self.thread.finished.connect(lambda: self.set_running(False))
        self.set_running(True)
        self.state.setText("Starting...")
        self.thread.start()

    def stop(self):
        if self.thread:
            self.thread.stop()
            self.state.setText("Stopping after current sweep...")

    def on_sweep(self, freqs, dbms, elapsed, peak_hz, peak_dbm):
        self.trace_curve.setData([f / 1e6 for f in freqs], dbms)
        self.t_hist.append(elapsed)
        self.p_hist.append(peak_dbm)
        self.hist_curve.setData(self.t_hist, self.p_hist)
        peak_mw = 10 ** (peak_dbm / 10)
        self.readout.setText(f"Peak: {peak_dbm:.2f} dBm ({fmt_power(peak_mw)}) at {peak_hz / 1e6:.4f} MHz")
        if self._run_measurement_name == "Single-tone calibration":
            metrics = distribution_metrics(freqs, dbms, self._run_source_freq_hz)
            if metrics:
                update_running_stats(self._single_tone_center_stats, metrics["center_bin_dbm"])
                update_running_stats(self._single_tone_leakage_stats, metrics["adjacent_leakage_pct"])
                sigma_center = running_sample_std(self._single_tone_center_stats)
                sigma_leakage = running_sample_std(self._single_tone_leakage_stats)
                if sigma_center is not None:
                    self._repeatability_text = (
                        f"Repeatability over {self._single_tone_center_stats[0]} sweeps: "
                        f"σ center-bin power {sigma_center:.4g} dB; "
                        f"σ adjacent leakage {sigma_leakage:.4g} percentage points.")
                else:
                    self._repeatability_text = "Repeatability σ requires at least two single-tone sweeps."
        self.update_bins(freqs, dbms)

    def load_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Target power profile", "", "Text (*.txt *.dat *.csv);;All (*)")
        if not path:
            return
        try:
            d = np.loadtxt(path)
            f, u = d[:, 1], d[:, 2]
        except Exception as e:
            self.bin_info.setText(f"Cannot read {path}: {e}")
            return
        self.target_f, self.target_u = f, u
        self.update_ur(None, None)
        # Match the measurement grid to the file so the bins line up.
        target_center = (f[0] + f[-1]) / 2
        if (self.measurement_mode.currentIndex() == 1
                or self.cw_measurement.currentIndex() == 0):
            self.source_freq.setValue(target_center)
        else:
            self.center.setValue(target_center)
        self.span.setValue((f[-1] - f[0]) * 1e3)
        self.points.setValue(len(f))
        self.redraw_bins()

    def clear_target(self):
        self.target_f = self.target_u = None
        self.target_curve.setVisible(False)
        self.update_ur(None, None)
        self.redraw_bins()

    def update_ur(self, meas_f, meas_mw):
        """Draw the profile on the R axis. R = -6..6 spans the grid linearly (R = 0 at its midpoint).

        With a target loaded, the measurement is scaled to the target's total; without one,
        the measured power per bin is drawn on its own.
        """
        have_meas = meas_f is not None and meas_mw is not None and meas_mw.sum() > 0
        has_target = self.target_u is not None
        if has_target:
            f0, f1 = self.target_f[0], self.target_f[-1]
        elif have_meas:
            f0, f1 = meas_f[0] / 1e6, meas_f[-1] / 1e6
        else:
            self.ur_target.setData([], [])
            self.ur_meas.setVisible(False)
            self.ur_plot.setLabel("left", "U(R)")
            self.ur_info.setText("Press Start to see the measured profile, "
                                 "or load a target profile on the Power bins tab.")
            return
        fc, per_r = (f0 + f1) / 2, (f1 - f0) / 12.0

        if has_target:
            self.ur_target.setData((self.target_f - fc) / per_r, self.target_u)
        else:
            self.ur_target.setData([], [])

        note = ""
        if have_meas:
            if has_target and self.target_u.sum() > 0:
                y = meas_mw * (self.target_u.sum() / meas_mw.sum())
                self.ur_plot.setLabel("left", "U(R)")
                note = " Measured curve is scaled to the target total."
            else:
                factor, unit = power_unit(float(meas_mw.max()))
                y = meas_mw * factor
                self.ur_plot.setLabel("left", f"Power per bin ({unit})")
            self.ur_meas.setData((meas_f / 1e6 - fc) / per_r, y)
            self.ur_meas.setVisible(True)
        else:
            self.ur_meas.setVisible(False)
            self.ur_plot.setLabel("left", "U(R)")
        self.ur_info.setText(f"{per_r * 1e3:.2f} kHz per unit R (R=0 at {fc:.4f} MHz)." + note)

    def reset_bins(self):
        self.bin_sum_mw = None
        self.bin_count = 0
        if self.bin_latest_mw is not None:
            self.bin_latest_mw = None
            self.bin_bars.setOpts(x=[0], height=[0], width=1)
            self.bin_info.setText("No data yet")

    def update_bins(self, freqs, dbms):
        mw = 10 ** (np.asarray(dbms, dtype=float) / 10)
        self.bin_freqs = np.asarray(freqs, dtype=float)
        self.bin_latest_mw = mw
        if self.bin_sum_mw is None or self.bin_sum_mw.shape != mw.shape:
            self.bin_sum_mw = np.zeros_like(mw)
            self.bin_count = 0
        self.bin_sum_mw += mw
        self.bin_count += 1
        self.redraw_bins()

    def draw_target(self, meas_f, meas_mw, units, x_idx):
        """Overlay the loaded target, scaled to the measured total (shape comparison)."""
        if self.target_u is None or units == 3:
            self.target_curve.setVisible(False)
            return None
        tgt = np.interp(meas_f / 1e6, self.target_f, self.target_u, left=0.0, right=0.0)
        share = tgt / tgt.sum() if tgt.sum() > 0 else tgt
        y = 100 * share if units == 2 else share * meas_mw.sum() * self._bin_factor
        x = np.arange(len(meas_f)) if x_idx else meas_f / 1e6
        self.target_curve.setData(x, y)
        self.target_curve.setVisible(True)
        return float(meas_mw[tgt > 0].sum() / meas_mw.sum()) if meas_mw.sum() > 0 else None

    def redraw_bins(self):
        if self.bin_latest_mw is None:
            self.update_resolution(None, None)
            if self.target_u is not None:  # no measurement yet: show the target alone
                self.bin_bars.setOpts(x=[0], height=[0], width=1)
                x_idx = self.bin_x.currentIndex() == 1
                x = np.arange(len(self.target_u)) if x_idx else self.target_f
                self.target_curve.setData(x, self.target_u)
                self.target_curve.setVisible(True)
                self.bin_plot.setLabel("left", "target U (relative)")
                self.bin_info.setText(f"Target loaded: {len(self.target_u)} bins, "
                                      f"{np.count_nonzero(self.target_u)} non-zero. Press Start to measure.")
            return
        mw = self.bin_latest_mw
        label = "latest sweep"
        if self.bin_mode.currentIndex() == 1:
            mw = self.bin_sum_mw / self.bin_count
            label = f"average of {self.bin_count} sweeps"
        # Each bin is a power in one RBW, and bins overlap when the step is smaller than the RBW.
        # Integrating needs sum * (step / RBW); the plain sum would over-count by RBW/step.
        step_hz = float(np.mean(np.diff(self.bin_freqs)))
        total_mw = float(mw.sum()) * step_hz / self._rbw_hz
        units = self.bin_units.currentIndex()
        auto_factor, auto_label = power_unit(float(mw.max()))
        if units == 0:
            self._bin_factor = auto_factor
            heights, ylabel = mw * auto_factor, auto_label
        elif units == 1:
            self._bin_factor = 1.0
            heights, ylabel = mw, "mW per bin"
        elif units == 2:
            self._bin_factor = 1.0
            heights, ylabel = 100 * mw / mw.sum(), "% of total"
        else:
            self._bin_factor = 1.0
            heights, ylabel = 10 * np.log10(mw), "dBm"
        x = self.bin_freqs / 1e6
        width = (x[-1] - x[0]) / max(len(x) - 1, 1)
        step_khz = width * 1e3
        if self.bin_x.currentIndex() == 1:
            x, width = np.arange(len(x), dtype=float), 1.0
            self.bin_plot.setLabel("bottom", "Bin index j")
        else:
            self.bin_plot.setLabel("bottom", "Frequency", "MHz")
        if units == 3:  # dBm is negative, so draw bars up from the bottom of the data
            base = float(heights.min()) - 2
            self.bin_bars.setOpts(x=x, y0=base, height=heights - base, width=width)
        else:
            self.bin_bars.setOpts(x=x, y0=0, height=heights, width=width)
        self.bin_plot.setLabel("left", ylabel)
        inside = self.draw_target(self.bin_freqs, mw, units, self.bin_x.currentIndex() == 1)
        self.update_ur(self.bin_freqs, mw)
        extra = f" In target bins: {100 * inside:.1f}% of power." if inside is not None else ""
        self.bin_info.setText(
            f"{len(x)} bins, {step_khz:.4f} kHz apart ({label}). "
            f"Total power (approx., sum x step/RBW): {fmt_power(total_mw)} "
            f"({10 * np.log10(max(total_mw, 1e-30)):.2f} dBm)." + extra
        )
        self.update_resolution(self.bin_freqs, mw)

    def update_resolution(self, freqs, mw):
        mode = self.measurement_mode.currentIndex()
        single_tone = mode == 0 and self.cw_measurement.currentIndex() == 0
        modulated = mode == 1
        if not single_tone and not modulated:
            self.resolution_curve.setData([], [])
            self.resolution_info.setText(
                "Coil-response sweeps measure power versus source frequency; bin-leakage metrics apply to a fixed tone.")
            return
        if freqs is None or mw is None or len(freqs) < 3:
            self.resolution_curve.setData([], [])
            self.resolution_info.setText("Start an acquisition to measure the frequency-bin distribution.")
            return

        freqs = np.asarray(freqs, dtype=float)
        mw = np.asarray(mw, dtype=float)
        tone_hz = self._run_source_freq_hz
        peak_dbm = float(np.max(10.0 * np.log10(np.maximum(mw, 1e-30))))
        offsets_khz = (freqs - tone_hz) / 1e3
        relative_db = 10.0 * np.log10(np.maximum(mw, 1e-30)) - peak_dbm
        floor_dbm = 10.0 * np.log10(np.maximum(mw, 1e-30))
        metrics = distribution_metrics(freqs, floor_dbm, tone_hz)
        edge_n = max(2, len(mw) // 10)
        edge_floor_mw = float(np.median(np.r_[mw[:edge_n], mw[-edge_n:]]))
        signal_mw = np.maximum(mw - edge_floor_mw, 0.0)
        if signal_mw.sum() <= 0:
            signal_mw = mw.copy()
        spacing_khz = metrics["bin_spacing_hz"] / 1e3
        samples_per_rbw = (self._rbw_hz / 1e3) / spacing_khz if spacing_khz > 0 else 0.0
        sampling_note = (" Sampling is coarse; increase bins or narrow the span to get at least 3 bins/RBW."
                         if samples_per_rbw < 3 else "")
        if single_tone:
            self.resolution_plot.setTitle("Single-tone leakage response by frequency bin")
            self.resolution_plot.setLabel("left", "Relative power", "dBc")
            self.resolution_curve.setData(offsets_khz, relative_db)
            center_mw = 10 ** (metrics["center_bin_dbm"] / 10.0)
            self.resolution_info.setText(
                f"RBW {self._rbw_hz / 1e3:g} kHz · bin spacing {spacing_khz:.4g} kHz · "
                f"center-bin power {metrics['center_bin_dbm']:.2f} dBm ({fmt_power(center_mw)}) · "
                f"center-bin share {metrics['center_bin_share_pct']:.2f}% · "
                f"adjacent-bin leakage {metrics['adjacent_leakage_pct']:.2f}% · "
                f"center ±1-bin share {metrics['center_pm1_share_pct']:.2f}% · "
                f"tone-to-bin offset {metrics['tone_to_bin_offset_hz'] / 1e3:.4g} kHz · "
                f"3 dB width {metrics['width_3db_hz'] / 1e3:.4g} kHz · "
                f"90% width {metrics['width_90_hz'] / 1e3:.4g} kHz. "
                "Shares and widths use power after subtracting the median edge-bin floor. "
                f"{self._repeatability_text}"
                f"{sampling_note}"
            )
        else:
            factor, unit = power_unit(float(signal_mw.max()))
            self.resolution_plot.setTitle("Modulated frequency power distribution")
            self.resolution_plot.setLabel("left", f"Power per bin ({unit})")
            self.resolution_curve.setData(offsets_khz, signal_mw * factor)
            peak_offset_khz = float(offsets_khz[int(np.argmax(signal_mw))])
            self.resolution_info.setText(
                f"{self.fm_waveform.currentText()} FM · RBW {self._rbw_hz / 1e3:g} kHz · "
                f"bin spacing {spacing_khz:.4g} kHz · peak offset {peak_offset_khz:.4g} kHz · "
                f"3 dB width {metrics['width_3db_hz'] / 1e3:.4g} kHz · "
                f"90% measured-power width {metrics['width_90_hz'] / 1e3:.4g} kHz. "
                "Distribution and width are estimated after subtracting the median edge-bin floor."
                f"{sampling_note}"
            )

    def on_failed(self, msg):
        self.state.setText(f"Error: {msg}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(20000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
