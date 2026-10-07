#!/usr/bin/env python3
import csv
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
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
        trace_f = None
        self.status.emit("Connecting...")
        try:
            ser = open_tinysa(c["port"] or None)
        except Exception as e:
            self.failed.emit(str(e))
            return

        try:
            command(ser, "pause")
            command(ser, f"rbw {c['rbw_hz'] / 1e3:g}")
            if c["trace_out"]:
                trace_f = open(c["trace_out"], "w", newline="")
                trace_w = csv.writer(trace_f)
                trace_w.writerow(["timestamp_utc", "sweep", "freq_hz", "dbm"])

            start_time = time.monotonic()
            n = 0
            bad = 0
            with open(c["out"], "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_utc", "elapsed_s", "peak_freq_hz", "peak_dbm", "peak_mw", "peak_uw"])
                f.flush()
                self.status.emit("Logging...")

                while not self._stop:
                    loop_start = time.monotonic()
                    try:
                        pts = acquire(ser, c["start_hz"], c["stop_hz"], c["points"])
                    except RuntimeError as e:
                        bad += 1
                        if bad >= 5:
                            raise RuntimeError(f"5 bad sweeps in a row, last error: {e}")
                        self.status.emit(f"Bad sweep ({e}), retrying...")
                        time.sleep(0.5)
                        continue
                    bad = 0

                    freqs = [p[0] for p in pts]
                    dbms = [p[1] for p in pts]
                    i = max(range(len(dbms)), key=dbms.__getitem__)
                    peak_hz, peak_dbm = freqs[i], dbms[i]
                    peak_mw = 10 ** (peak_dbm / 10)

                    now = datetime.now(timezone.utc).isoformat()
                    elapsed = time.monotonic() - start_time
                    w.writerow([now, f"{elapsed:.3f}", f"{peak_hz:.1f}", f"{peak_dbm:.3f}",
                                f"{peak_mw:.6g}", f"{peak_mw * 1000:.6g}"])
                    f.flush()
                    if trace_f:
                        trace_w.writerows([now, n, f"{fr:.1f}", f"{d:.3f}"] for fr, d in pts)
                        trace_f.flush()
                    n += 1

                    self.sweep.emit(freqs, dbms, elapsed, peak_hz, peak_dbm)

                    if c["duration"] and elapsed >= c["duration"]:
                        break
                    while not self._stop and time.monotonic() - loop_start < c["interval"]:
                        time.sleep(0.05)
        except Exception as e:
            self._failed = True
            self.failed.emit(str(e))
        finally:
            if trace_f:
                trace_f.close()
            try:
                command(ser, "resume")
            except Exception:
                pass
            ser.close()
            if not self._failed:  # keep the error message visible if the run failed
                self.status.emit("Stopped")


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

        # ---- settings form ----
        self.port = QComboBox()
        self.port.setEditable(True)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_ports)
        port_row = QHBoxLayout()
        port_row.addWidget(self.port, 1)
        port_row.addWidget(refresh)

        self.center = self._spin(0.1, 6000, 213.0, 4, " MHz")
        self.span = self._spin(1, 6e6, 400.0, 1, " kHz")
        self.rbw = self._spin(0.2, 850, 10.0, 1, " kHz")
        self.points = QSpinBox()
        self.points.setRange(11, 1000)
        self.points.setValue(500)
        self.interval = self._spin(0, 3600, 1.0, 1, " s")
        self.duration = self._spin(0, 86400, 0.0, 0, " s (0 = until Stop)")

        preset = QPushButton("NMR-AI grid (proton, 500 bins)")
        preset.clicked.connect(self.apply_nmr_ai_preset)

        self.out = QLineEdit("tinysa_power_log.csv")
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out, 1)
        out_row.addWidget(browse)

        self.save_traces = QCheckBox("Also save full traces to")
        self.trace_out = QLineEdit("tinysa_traces.csv")
        trace_row = QHBoxLayout()
        trace_row.addWidget(self.save_traces)
        trace_row.addWidget(self.trace_out, 1)

        form = QFormLayout()
        form.addRow("Port", port_row)
        form.addRow("Center", self.center)
        form.addRow("Span", self.span)
        form.addRow("RBW", self.rbw)
        form.addRow("Bins / points", self.points)
        form.addRow("Interval", self.interval)
        form.addRow("Duration", self.duration)
        form.addRow(preset)
        form.addRow("Output CSV", out_row)
        form.addRow(trace_row)
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
        left_w.setFixedWidth(360)

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
        self.bin_units.addItems(["power per bin (auto unit)", "% of total power", "dBm per bin"])
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

        self._bin_factor = 1e3
        self._rbw_hz = 10e3

        tabs = QTabWidget()
        tabs.addTab(trace_tab, "Trace")
        tabs.addTab(bins_tab, "Power bins")
        tabs.addTab(ur_tab, "U(R) profile")

        root = QHBoxLayout()
        root.addWidget(left_w)
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

    def apply_nmr_ai_preset(self):
        self.center.setValue(PROTON_CENTER_MHZ)
        self.span.setValue((NMR_AI_BINS - 1) * NMR_AI_STEP_KHZ)
        self.points.setValue(NMR_AI_BINS)
        self.rbw.setValue(3.0)

    def refresh_ports(self):
        self.port.clear()
        self.port.addItems(find_ports())

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Output CSV", self.out.text(), "CSV (*.csv)")
        if path:
            self.out.setText(path)

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def start(self):
        if not self.port.currentText().strip():
            self.refresh_ports()
        port = self.port.currentText().strip()  # empty means auto-detect when the logger starts
        center, span = self.center.value() * 1e6, self.span.value() * 1e3
        cfg = {
            "port": port,
            "start_hz": center - span / 2,
            "stop_hz": center + span / 2,
            "rbw_hz": self.rbw.value() * 1e3,
            "points": self.points.value(),
            "interval": self.interval.value(),
            "duration": self.duration.value(),
            "out": self.out.text().strip() or "tinysa_power_log.csv",
            "trace_out": self.trace_out.text().strip() if self.save_traces.isChecked() else "",
        }
        self.t_hist, self.p_hist = [], []
        self.hist_curve.setData([], [])
        self.reset_bins()
        self._rbw_hz = cfg["rbw_hz"]
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
        self.center.setValue((f[0] + f[-1]) / 2)
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
        if self.target_u is None or units == 2:
            self.target_curve.setVisible(False)
            return None
        tgt = np.interp(meas_f / 1e6, self.target_f, self.target_u, left=0.0, right=0.0)
        share = tgt / tgt.sum() if tgt.sum() > 0 else tgt
        y = 100 * share if units == 1 else share * meas_mw.sum() * self._bin_factor
        x = np.arange(len(meas_f)) if x_idx else meas_f / 1e6
        self.target_curve.setData(x, y)
        self.target_curve.setVisible(True)
        return float(meas_mw[tgt > 0].sum() / meas_mw.sum()) if meas_mw.sum() > 0 else None

    def redraw_bins(self):
        if self.bin_latest_mw is None:
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
        self._bin_factor, unit_label = power_unit(float(mw.max()))
        if units == 0:
            heights, ylabel = mw * self._bin_factor, unit_label
        elif units == 1:
            heights, ylabel = 100 * mw / mw.sum(), "% of total"
        else:
            heights, ylabel = 10 * np.log10(mw), "dBm"
        x = self.bin_freqs / 1e6
        width = (x[-1] - x[0]) / max(len(x) - 1, 1)
        step_khz = width * 1e3
        if self.bin_x.currentIndex() == 1:
            x, width = np.arange(len(x), dtype=float), 1.0
            self.bin_plot.setLabel("bottom", "Bin index j")
        else:
            self.bin_plot.setLabel("bottom", "Frequency", "MHz")
        if units == 2:  # dBm is negative, so draw bars up from the bottom of the data
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
