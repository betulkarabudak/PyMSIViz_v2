# ---------------- Standard library ----------------
import sys
import os
import time
import csv
import tracemalloc
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
# ---------------- Scientific stack ----------------
import numpy as np    
try:
    from scipy.stats import (
        gaussian_kde, norm,
        mannwhitneyu,
        kruskal, f_oneway, levene, shapiro
    )
    _HAS_SCIPY_STATS = True
except Exception:
    _HAS_SCIPY_STATS = False

try:
    from scipy.signal import find_peaks, savgol_filter
    _HAS_SCIPY_SIGNAL = True
except Exception:
    _HAS_SCIPY_SIGNAL = False
    
# ---------------- PyQt5 ----------------
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QProgressBar, QStatusBar,
    QMenuBar, QAction, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QTabWidget,
    QSplitter, QScrollArea, QCheckBox,
    QListWidget, QAbstractItemView, QInputDialog,
    QListView, QStyledItemDelegate)
# ---------------- Matplotlib ----------------
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
# ---------------- App / Core ----------------
from core.version import __version__
from core.logger import logger
from core.styles import get_app_stylesheet
from core.threads import LoadImzMLThread
from core.utils import ( getionimage, crop_zero_rows_cols, validate_numeric_input, bisect_spectrum)
from core.mpl_widgets import MplWidget
from core.roi_mgr import ROIManager
from workers.cancelable_worker import CancelableTaskWorker
from core.msi_tasks import (task_ai_assist_top_peak,task_compute_ionimage,task_roi_compare_violin,task_roi_volcano)
from core.analytics import peak_pick_features, benjamini_hochberg,compute_global_spectrum_binned
from workers.worker_manager import WorkerManager
from pyimzml.ImzMLParser import ImzMLParser
# ---------------- UI builders ----------------
from ui.builders import ( build_group_datafile, build_group_mz, build_group_actions,
                           build_group_roi, build_menu)

class FixedHeightDelegate(QStyledItemDelegate):
    def __init__(self, h=28, parent=None):
        super().__init__(parent)
        self.h = h

    def sizeHint(self, option, index):
        sz = super().sizeHint(option, index)
        sz.setHeight(self.h)
        return sz
# --------------------------- Main Window ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
# =========== Core / Data State =========================
        self.imzml_filename = None
        self.parser = None
        self.large_dataset_mode = False
        self.roi_mgr = ROIManager()
        self.roi_selector = None
        self.roi_selector_type = None  # "rect" / "lasso"
        self._active_peak_mz = None
# =========== Caching ===================================
        self._ion_cache = {}
        self._ion_cache_order = []
        self._ion_cache_max = 8
        self._feature_cache = None       # np.array of mz
        self._feature_cache_key = None   # (n_bins, max_pixels, top_k, min_mz_dist, snr)
        self._roi_value_cache = {}
        self.analysis_cache = {}
        self.datasets = {}  
        self.active_dataset_id = None
        self._roi_counter = 1  
# =========== Plot / ROI State ===========================
        self.plot_state = {"box": False,"violin": False,"volcano": False}
        self.plot_lock  = {"box": False,"violin": False,"volcano": False}
        self.roi_dirty  = False
# =========== Benchmark / Performance ====================
        self.enable_benchmark = True
        self.benchmark_log = []
        self.last_total_size_mb = None
        self.last_load_time_s = None
        self.last_peak_ram_mb = None
        self.last_image1_render_time_s = None
        self.last_image2_render_time_s = None
        self.last_image3_render_time_s = None
        self.last_multi_render_total_s = None
# =========== UI / Window ================================
        self.setWindowTitle("PyMSIViz v2 - Responsive + ROI + AI Assist")
        self.resize(1200, 780)
        self.setMinimumSize(980, 700)
        self.setStyleSheet(get_app_stylesheet())
        self.chkTopPeaks = False
# =========== UI Construction ============================
        self._build_ui()
        build_menu(self)
        # UI builders (bound methods)
        self._build_group_datafile = build_group_datafile.__get__(self)
        self._build_group_mz       = build_group_mz.__get__(self)
        self._build_group_actions = build_group_actions.__get__(self)
        self._build_group_roi     = build_group_roi.__get__(self)
        self._build_menu          = build_menu.__get__(self)
        self.buttonBox.setEnabled(False)
# =========== Status bar =================================
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("🚀 Ready")
# =========== Signals / Runtime ==========================
        self._connect_signals()
        self._is_closing = False
        self._close_force_after_ms = 1200  # hızlı kapanış
        self._set_multi_enabled(False)     # default: OFF
        self._running_worker = None
        self._global_spectrum_cache = {}
# =========== Workers ====================================
        self.worker_mgr = WorkerManager()
        self._set_ai_assist_enabled(False)
# =========== Debug / Post-init fixes ====================
        print("plotTabs enabled:", self.plotTabs.isEnabled())
        print("plotTabs count:", self.plotTabs.count())
        for i in range(self.plotTabs.count()):
            print(i, self.plotTabs.tabText(i), self.plotTabs.isTabEnabled(i))
        QTimer.singleShot(0, self._force_enable_plots)
        QTimer.singleShot(0, self._lock_plot_tabs_enabled)
# ===================== UI HELPERS =======================
    def _format_bytes(self, size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"

    def _on_norm_changed(self):
        if hasattr(self, "_ion_image_cache"):
            self._ion_image_cache.clear()
        self.statusbar.showMessage("🔄 Normalization changed (cache cleared)")
    
    def _force_enable_plots(self):
        print("Forcing plot tabs enable")

    
    def _lock_plot_tabs_enabled(self):
        for i in range(self.plotTabs.count()):
            self.plotTabs.setTabEnabled(i, True)
            
 # --------------------------- Multi UI logic ---------------------------
    def _set_multi_enabled(self, enabled: bool):
        # ---------------- Images (Image2/Image3) ----------------
        try:
            self.imageTabs.setTabVisible(1, enabled)
            self.imageTabs.setTabVisible(2, enabled)
        except Exception:
            self.imageTabs.widget(1).setVisible(enabled)
            self.imageTabs.widget(2).setVisible(enabled)            
# ---------------- mz2-mz3 satırlarını hide/show ----------------
        for w in self._multi_row_widgets:
            w.setVisible(enabled)
            if not enabled and isinstance(w, QLineEdit):
                w.clear()    
# Multi kapalıyken Image tab'ında 2/3'te kalmasın
        if not enabled and self.imageTabs.currentIndex() in (1, 2):
            self.imageTabs.setCurrentIndex(0)
        self._sync_multi_visibility()

    def _sync_multi_visibility(self):
        enabled = self.multiCheck.isChecked()
    
        mz2_ok = bool(self.targetB_textbox.text().strip() and self.width_textboxb.text().strip())
        mz3_ok = bool(self.targetC_textbox.text().strip() and self.width_textboxc.text().strip())
    
        show_img2 = enabled and mz2_ok
        show_img3 = enabled and mz2_ok and mz3_ok
    
        self.imageTabs.setTabVisible(1, show_img2)
        self.imageTabs.setTabVisible(2, show_img3)

# --- IMAGE tabs ---
        show_img2 = enabled and mz2_ok
        show_img3 = enabled and mz2_ok and mz3_ok    
        try:
            self.imageTabs.setTabVisible(1, show_img2)
            self.imageTabs.setTabVisible(2, show_img3)
        except Exception:
            self.imageTabs.widget(1).setVisible(show_img2)
            self.imageTabs.widget(2).setVisible(show_img3)

        if self.leftTabs.currentIndex() == 1:
            if (not show_img2 and self.imageTabs.currentIndex() == 1) or (not show_img3 and self.imageTabs.currentIndex() == 2):
                self.imageTabs.setCurrentIndex(0)

        if self.leftTabs.currentIndex() == 2:
            cur = self.plotTabs.currentIndex()
            if (cur == 1 and (not (enabled and mz2_ok))) or (cur == 2 and (not (enabled and mz2_ok and mz3_ok))):
                self.plotTabs.setCurrentIndex(0)            

    def _on_worker_cancelled(self):
        self.statusbar.showMessage("⛔ Operation cancelled")
        self._running_worker = None
        
    def _on_worker_finished(self, payload):
        if payload is None:
            self.statusbar.showMessage("⛔ Operation cancelled")
            self._running_worker = None
            return
   
        self._running_worker = None
      
    def _on_worker_error(self, msg):
        QMessageBox.critical(self, "❌ Task Error", msg)
        self.statusbar.showMessage("❌ Task failed")
        self._running_worker = None

    def cancel_running_task(self):
        if self._running_worker is not None:
            self._running_worker.cancel()
            self.statusbar.showMessage("⏹️ Cancelling…")
            
    def ask_quit(self):
        reply = QMessageBox.question(
            self,
            "Exit",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
    
        if reply == QMessageBox.Yes:
            # çalışan worker varsa iptal et
            if self._running_worker is not None:
                try:
                    self._running_worker.cancel()
                except Exception:
                    pass

            self.close()

    def update_plots_like_images(self, kind="box"):
        if not self._check_loaded():
            return

        self.leftTabs.setCurrentIndex(2)     # Plot ana tab
        self.plotTabs.setCurrentIndex(0)     # Plot 1
        QApplication.processEvents()
    
        mz1, tol1, mz2, tol2, mz3, tol3, z = self._read_mz_block()
        d1 = self._get_nonzero_flat(mz1, tol1, z)   
        self._draw_dist_on(
            self.boxPlot,
            d1,
            f"Plot 1 ({kind.title()}): m/z {mz1:.4f} ± {tol1}",
            kind=kind
        )

        if self.multiCheck.isChecked() and mz2 is not None and tol2 is not None:
    
            self.plotTabs.setCurrentIndex(1)
            QApplication.processEvents()
    
            data = [d1]
            labels = [f"{mz1:.4f}"]
    
            d2 = self._get_nonzero_flat(mz2, tol2, z)
            data.append(d2)
            labels.append(f"{mz2:.4f}")
    
            if mz3 is not None and tol3 is not None:
                d3 = self._get_nonzero_flat(mz3, tol3, z)
                data.append(d3)
                labels.append(f"{mz3:.4f}")
    
            ax = self.multiPlot.canvas.ax
            ax.clear()
    
            if kind == "violin":
                parts = ax.violinplot(
                    data,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("#93c5fd")
                    pc.set_edgecolor("#2563eb")
                    pc.set_alpha(0.85)
            else:
                box = ax.boxplot(
                    data,
                    labels=labels,
                    patch_artist=True,
                    showfliers=False
                )
                for b in box["boxes"]:
                    b.set_facecolor("#93c5fd")
                    b.set_edgecolor("#2563eb")
                for m in box["medians"]:
                    m.set_color("#111827")
                    m.set_linewidth(2)
    
            self._apply_axes_style(
                ax,
                title="Plot 2 (Multi m/z)",
                ylabel="Intensity"
            )
    
            self.multiPlot.canvas.draw_idle()

        for i in range(self.plotTabs.count()):
            self.plotTabs.setTabEnabled(i, True)
    
        self.leftTabs.setCurrentIndex(2)
        self.plotTabs.setCurrentIndex(0)
        self.boxPlot.canvas.draw_idle()


    def _update_plot_button_state(self):
        try:
            mz_ok  = bool(self.targetA_textbox.text().strip())
            tol_ok = bool(self.width_textbox.text().strip())
            z_ok   = bool(self.z_textbox.text().strip())
    
            enable = mz_ok and tol_ok and z_ok
            self.buttonBox.setEnabled(enable)
    
        except Exception:
            self.buttonBox.setEnabled(False)
 
    def _update_image_processing_state(self):
        cmap_ok = bool(self.cb.currentText().strip())
    
        # ---- base (Image 1) ----
        mz1_ok = bool(self.targetA_textbox.text().strip())
        tol1_ok = bool(self.width_textbox.text().strip())
        z_ok = bool(self.z_textbox.text().strip())
        base_ok = mz1_ok and tol1_ok and z_ok

        if self.multiCheck.isChecked():
            mz2_ok = bool(self.targetB_textbox.text().strip())
            tol2_ok = bool(self.width_textboxb.text().strip())
    
            mz3_txt = self.targetC_textbox.text().strip()
            tol3_txt = self.width_textboxc.text().strip()
            mz3_ok = (not mz3_txt) or bool(tol3_txt)
    
            multi_ok = mz2_ok and tol2_ok and mz3_ok
        else:
            multi_ok = True
    
        enable = cmap_ok and base_ok and multi_ok
        self.buttonImage.setEnabled(enable)

    def _ui_ai_assist_top_peak(self):
        if not self._check_loaded():
            return 
        self.statusbar.showMessage("AI Assist: finding top peak...")   
        max_pixels = 800 if self.large_dataset_mode else 3000        
        worker = CancelableTaskWorker(
            task_ai_assist_top_peak,
            self.parser,
            max_pixels=max_pixels,
            n_bins=6000)
        def _on_result(top_mz):
            if top_mz is None:
                QMessageBox.warning(self, "AI Assist", "No peak found.")
                return
                
            self.targetA_textbox.setText(f"{top_mz:.4f}")

            if not self.width_textbox.text().strip():
                self.width_textbox.setText("0.5")
            if not self.z_textbox.text().strip():
                self.z_textbox.setText("1")
    
            self.statusbar.showMessage(f"✅ Top peak found: m/z {top_mz:.4f}")
            self.process_image1_only()

        def _on_error(msg):
            QMessageBox.critical(self, "AI Assist Error", msg)
            self.statusbar.showMessage("❌ AI Assist failed")
    
        worker.result.connect(_on_result)
        worker.error.connect(_on_error)   
        self.worker_mgr.start(worker)

    def _fix_combo_popup(self, combo: QComboBox):
        popup_css = """
            QListView {
                background: #ffffff;
                color: #111827;
                border: 1px solid #d1d5db;
                outline: 0;}
            QListView::item { padding: 6px 10px; }
            QListView::item:hover { background: #e0f2fe; color: #111827; }
            QListView::item:selected:active,
            QListView::item:selected:!active { background: #1f7ae0; color: #ffffff; }
        """
        v = QListView()
        v.setFocusPolicy(Qt.NoFocus)
        v.setStyleSheet(popup_css)
    
        v.setItemDelegate(FixedHeightDelegate(h=26, parent=v))
        v.setUniformItemSizes(True)    
        combo.setView(v)

    def _clear_feature_cache(self):
        self._feature_cache = None
        self._feature_cache_key = None
    
    def _get_selected_features(self, *,
                               max_pixels=600,
                               n_bins=2500,
                               top_k=400,
                               min_mz_dist=0.20,
                               snr=3.0,):        
        if not self._check_loaded():
            return np.array([], dtype=float)
        key = (int(n_bins), int(max_pixels), int(top_k),float(min_mz_dist), float(snr))
        if self._feature_cache is not None and self._feature_cache_key == key:
            return self._feature_cache
    
        df_mz, df_int = self._compute_global_spectrum_binned(max_pixels=int(max_pixels),n_bins=int(n_bins), stat="sum")    
        feats = self._peak_pick_features(df_mz, df_int, top_k=int(top_k),
                                         min_mz_dist=float(min_mz_dist),
                                         snr=float(snr))
        if feats.size < 30:
            top_idx = np.argsort(df_int)[-min(300, df_mz.size):]
            feats = np.unique(np.sort(df_mz[top_idx].astype(float)))
    
        self._feature_cache = feats
        self._feature_cache_key = key
        return feats 
 
    
 
    def _format_roi_stats_line(
        self,
        mz,
        tol,
        valsA,
        valsB=None,
        p_val=None,
        delta=None
    ):
        valsA = np.asarray(valsA, dtype=float)
        valsA = valsA[np.isfinite(valsA)]
    
        n = valsA.size
        if n == 0:
            return f"m/z {mz:.4f} ± {tol} | n=0"
    
        mean = np.mean(valsA)
        median = np.median(valsA)
        std = np.std(valsA, ddof=1) if n > 1 else 0.0
        vmin = np.min(valsA)
        vmax = np.max(valsA)
        vsum = np.sum(valsA)
    
        line = (
            f"m/z {mz:.4f} ± {tol} | "
            f"n={n} | "
            f"mean={mean:.4f} | "
            f"median={median:.4f} | "
            f"std={std:.5f} | "
            f"min={vmin:.5f} | "
            f"max={vmax:.5f} | "
            f"sum={vsum:.2f}"
        )
    
        if p_val is not None:
            line += f" | MWU p={p_val:.2e}"
    
        if delta is not None:
            line += f" | Cliff’s δ={delta:.2f}"
    
        return line
    
 
    
 
    def _reset_roi_selectors(self):
        """Clear & re-seed ROI A/B selectors so they never carry stale items."""
        if hasattr(self, "comboROI_A"):
            self.comboROI_A.blockSignals(True)
            self.comboROI_A.clear()
            self.comboROI_A.blockSignals(False)
        if hasattr(self, "comboROI_B"):
            self.comboROI_B.blockSignals(True)
            self.comboROI_B.clear()
            self.comboROI_B.blockSignals(False)    
    
    def _mw_plot_roi_violin(self):
        QMessageBox.information(
            self,
            "Deprecated",
            "This exploratory violin plot is deprecated. Use ROI Compare Violin instead.")
        return
      
    def _build_ui(self):
        self._build_root_layout()
        self._build_left_tabs()
        self._build_plot_tabs()
        self._build_right_panel()
        self._finalize_splitter()

    def _build_root_layout(self):
        central = QWidget(self)
        self.setCentralWidget(central)
    
        self.rootLayout = QHBoxLayout(central)
        self.rootLayout.setContentsMargins(10, 10, 10, 10)
        self.rootLayout.setSpacing(10)
    
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.rootLayout.addWidget(self.splitter)

    def _build_spectrum_tab(self):
        self.tabSpectrum = QWidget()
        v1 = QVBoxLayout(self.tabSpectrum)
        v1.setContentsMargins(6, 6, 6, 6)
    
        self.leftTabs.addTab(self.tabSpectrum, "Spectrum")     
        # ================= Spectrum Plot =================
        self.spectrumPlot = MplWidget()
        self.spectrumPlot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        v1.addWidget(self.spectrumPlot, 1)

        self.aiAssistBtn = QPushButton("AI Assist: Find Top Peak")


    def _build_image_tab(self):
        self.tabImage = QWidget()
        v2 = QVBoxLayout(self.tabImage)
        v2.setContentsMargins(6, 6, 6, 6)
    
        self.imageTabs = QTabWidget()
        self.imageTabs.setDocumentMode(True)
    
        self.image1Plot = MplWidget()
        self.image2Plot = MplWidget()
        self.image3Plot = MplWidget()
    
        self.imageTabs.addTab(self.image1Plot, "Image 1")
        self.imageTabs.addTab(self.image2Plot, "Image 2")
        self.imageTabs.addTab(self.image3Plot, "Image 3")
        
        self.leftTabs.addTab(self.tabImage, "Image")    
        self.imageTabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding )
        v2.addWidget(self.imageTabs, 1)
        
    def _build_plot_tabs(self):
        self.tabPlot = QWidget()
        v3 = QVBoxLayout(self.tabPlot)
        v3.setContentsMargins(6, 6, 6, 6)
    
        self.plotTabs = QTabWidget()
        self.plotTabs.setDocumentMode(True)
    
        self.boxPlot = MplWidget()
        self.multiPlot = MplWidget()
        self.volcanoPlot = MplWidget()
        self.posthocPlot = MplWidget()
    
        self.plotTabs.addTab(self.boxPlot, "Plot 1")
        self.plotTabs.addTab(self.multiPlot, "Plot 2")
        self.plotTabs.addTab(self.posthocPlot, "Plot 3")        
        self.plotTabs.addTab(self.volcanoPlot, "Plot 4")

        self.plotTabs.setTabEnabled(2, False)
        self.plotTabs.setTabEnabled(3, False)
        self.leftTabs.addTab(self.tabPlot, "Plot")    
        self.plotTabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        v3.addWidget(self.plotTabs, 1)  # 🔥


        
    def _build_left_tabs(self):
        self.leftTabs = QTabWidget()
        self.leftTabs.setDocumentMode(True)
        self.leftTabs.setMovable(False)
        self.leftTabs.setObjectName("leftPanel")
        # Spectrum + Image tabları
        self._build_spectrum_tab()
        self._build_image_tab()
    
        self.splitter.addWidget(self.leftTabs)
        self.leftTabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.leftTabs.setStyleSheet("""
                                    QTabWidget::pane {
                                        border: none;
                                        background: white;
                                        top: 0px;
                                    }
                                    
                                    QTabBar {
                                        background: white;
                                    }

                                    QTabWidget {
                                        background: white;
                                    }
                                    """)

       
    def _build_right_panel(self):
        self.rightScroll = QScrollArea()
        self.rightScroll.setWidgetResizable(True)
        self.rightScroll.setFrameShape(QFrame.NoFrame)
        self.rightScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rightScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.rightScroll.setFixedWidth(330)
    
        self.rightWidget = QWidget()
        self.rightScroll.setWidget(self.rightWidget)
        self.rightWidget.setObjectName("rightPanel")
        self.rightLayout = QVBoxLayout(self.rightWidget)
        self.rightLayout.setContentsMargins(8, 8, 8, 8)
        self.rightLayout.setSpacing(8)
    
        build_group_datafile(self)
        build_group_mz(self)
        build_group_actions(self)
        build_group_roi(self)
    
        self.rightLayout.addWidget(self.groupData)
        self.rightLayout.addWidget(self.groupMz)
        self.rightLayout.addWidget(self.groupActions)
        self.rightLayout.addWidget(self.groupROI)
    
        self.splitter.addWidget(self.rightScroll)

    def _finalize_splitter(self):
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([900, 300])
       
    def _connect_signals(self):
        self._connect_data_signals()
        self._connect_spectrum_signals()
        self._connect_image_signals()
        self._connect_plot_signals()
        self._connect_roi_signals()
        self._connect_misc_signals()
        
    def _connect_data_signals(self):
        self.buttonLoadData.clicked.connect(self.load_file)
        self.loadAction.triggered.connect(self.load_file)

    
    def _connect_spectrum_signals(self):
        self.buttonSpectrum.clicked.connect(self.process_spectrum)
        self.spectrumAction.triggered.connect(self.process_spectrum)
        self.aiAction.triggered.connect(self.ai_assist_top_peak_image)
        
    def _connect_image_signals(self):
        self.chkTopPeaks.toggled.connect(self._on_top_peaks_toggled)
        self.targetA_textbox.textChanged.connect(self._update_image_processing_state)
        self.width_textbox.textChanged.connect(self._update_image_processing_state)
        self.z_textbox.textChanged.connect(self._update_image_processing_state)
        self.aiAssistBtn.clicked.connect(self.ai_assist_top_peak_image)   
        self.buttonImage.clicked.connect(self.on_image_processing_clicked)
        self.targetA_textbox.textChanged.connect(self._update_plot_button_state)
        self.width_textbox.textChanged.connect(self._update_plot_button_state)
        self.z_textbox.textChanged.connect(self._update_plot_button_state)


    def _connect_plot_signals(self):
        self.buttonBox.clicked.connect(self.run_selected_plot)

    def _connect_roi_signals(self):
        self.btnROIrect.clicked.connect(self.start_roi_rectangle)
        self.btnROIlasso.clicked.connect(self.start_roi_lasso)
        self.btnROIclear.clicked.connect(self.clear_roi)
        self.btnROIspectrum.clicked.connect(self.roi_mean_spectrum)
    
        self.btnROIsave.clicked.connect(self.save_current_roi)
        self.btnROIstats.clicked.connect(self.roi_intensity_stats)
        self.btnROIoverlay.clicked.connect(self.toggle_roi_mask_overlay)
        self.btnROISimilarity.clicked.connect(self.roi_similarity)
        self.btnROIcompare.clicked.connect(self.roi_compare_boxplot)
        self.btnROIcsv.clicked.connect(self.export_roi_csv)
        self.btnROIcmpV.clicked.connect(self.roi_compare_violin)
        self.btnVolcano.clicked.connect(self._roi_volcano_threaded)
        self.comboROI_A.currentIndexChanged.connect(self._on_roi_ab_changed)
        self.comboROI_B.currentIndexChanged.connect(self._on_roi_ab_changed)
    
    def _connect_misc_signals(self):
        self.multiCheck.toggled.connect(self._set_multi_enabled)
        self.aboutAction.triggered.connect(self.show_about)
        self.benchExportAction.triggered.connect(self.export_benchmark_csv)
        self.benchClearAction.triggered.connect(self.clear_benchmark_log)
    
        self.targetB_textbox.textChanged.connect(self._sync_multi_visibility)
        self.width_textboxb.textChanged.connect(self._sync_multi_visibility)
        self.targetC_textbox.textChanged.connect(self._sync_multi_visibility)
        self.width_textboxc.textChanged.connect(self._sync_multi_visibility)
        # ---- Image Processing state sync ----
        self.targetA_textbox.textChanged.connect(self._update_image_processing_state)
        self.width_textbox.textChanged.connect(self._update_image_processing_state)
        self.z_textbox.textChanged.connect(self._update_image_processing_state)
        
        self.targetB_textbox.textChanged.connect(self._update_image_processing_state)
        self.width_textboxb.textChanged.connect(self._update_image_processing_state)
        self.targetC_textbox.textChanged.connect(self._update_image_processing_state)
        self.width_textboxc.textChanged.connect(self._update_image_processing_state)
        self.btnMultiROIStats.clicked.connect(self.run_multi_roi_statistics)
        self.multiCheck.toggled.connect(self._update_image_processing_state)
        self.buttonBox.clicked.connect(self.run_selected_plot)
        self.cb.currentIndexChanged.connect(self._update_image_processing_state)
        self.cb2.currentIndexChanged.connect(self._update_image_processing_state)
        #self.btnCancel.clicked.connect(self.cancel_running_task)
        self.clearButton.clicked.connect(self.clear_all)
        self.closeButton.clicked.connect(self.ask_quit)
        
        
    def _disable_navigation(self):
        """
        Disable matplotlib navigation (zoom / pan) before ROI actions.
        """
        if hasattr(self, "spectrumPlot") and hasattr(self.spectrumPlot, "toolbar"):
            self.spectrumPlot.toolbar.pan(False)
            self.spectrumPlot.toolbar.zoom(False)
    
        if hasattr(self, "imagePlot") and hasattr(self.imagePlot, "toolbar"):
            self.imagePlot.toolbar.pan(False)
            self.imagePlot.toolbar.zoom(False)
     

    def _on_roi_ab_changed(self):
        self.roi_dirty = True

    def run_selected_plot(self):
        kind = self.plotTypeCombo.currentText().strip()
    
        if kind == "Box Plot":
            self.plot_single_mz(kind="box")
        elif kind == "Violin Plot":
            self.plot_single_mz(kind="violin")

    def plot_single_mz(self, kind="box"):
        # 1️⃣ Dosya yüklü mü
        if not self._check_loaded():
            return
    
        # 2️⃣ m/z – tol – z oku
        try:
            mz  = float(self.targetA_textbox.text())
            tol = float(self.width_textbox.text())
            z   = int(self.z_textbox.text())
        except Exception:
            QMessageBox.warning(self, "Plot", "Please enter valid m/z, tolerance and z.")
            return
    
        # 3️⃣ Ion image → intensity vektörü
        img = self._get_ion_image_cached(mz, tol, z)
    
        vals = img.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
    
        if vals.size < 10:
            QMessageBox.information(self, "Plot", "Not enough intensity values.")
            return
    
        # 4️⃣ Plot-1 canvas
        ax = self.boxPlot.canvas.ax
        ax.clear()
    
        if kind == "violin":
            parts = ax.violinplot(
                [vals],
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#93c5fd")
                pc.set_edgecolor("#2563eb")
                pc.set_alpha(0.85)
        else:
            box = ax.boxplot(
                [vals],
                patch_artist=True,
                showfliers=False
            )
            for b in box["boxes"]:
                b.set_facecolor("#93c5fd")
                b.set_edgecolor("#2563eb")
    
        self._apply_axes_style(
            ax,
            title=f"{kind.title()} | m/z {mz:.4f} ± {tol} (z={z})",
            ylabel="Intensity"
        )
    
        self.boxPlot.canvas.draw_idle()
        self.leftTabs.setCurrentIndex(2)
        self.plotTabs.setCurrentIndex(0)

    
    DELTA_COLORS = {
        "negligible": "#9ca3af",  # gri
        "small": "#2563eb",       # mavi
        "medium": "#f59e0b",      # turuncu
        "large": "#dc2626"        # kırmızı
    }

    def _delta_category(self, delta):
        d = abs(delta)
        if d < 0.147:
            return "negligible"
        elif d < 0.33:
            return "small"
        elif d < 0.474:
            return "medium"
        else:
            return "large"






    def _on_top_peaks_toggled(self, checked):
        self.enable_top_peaks = bool(checked)
    
        if checked:
            self.statusbar.showMessage("🧠 Top peak detection ENABLED")
        else:
            self.statusbar.showMessage("⛔ Top peak detection DISABLED")

    PLOT_FONT = {
        "title": 9,
        "label": 8,
        "tick": 8,
        "legend": 7,
        "subtitle": 7,
        "stat": 7
    }



    def _add_mz_subtitle(self, ax, text):
        # ince ayırıcı çizgi
        ax.plot(
            [0.15, 0.85], [0.965, 0.965],
            transform=ax.transAxes,
            color="#d1d5db",
            lw=0.8
        )
    
        ax.text(
            0.5, 0.945,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=self.PLOT_FONT["subtitle"],
            style="italic",
            color="#4b5563",
            clip_on=False
        )

    
    def run_multi_roi_statistics(self):
        if not self._check_loaded():
            return
    
        selected = [i.text() for i in self.roiList.selectedItems()]
        if len(selected) < 3:
            QMessageBox.warning(
                self,
                "Multi-ROI Statistics",
                "Please select at least 3 ROIs."
            )
            return
    
        mz1, tol1, _, _, _, _, _ = self._read_mz_block()
    
        try:
            data = []
            labels = []
    
            for name in selected:
                info = self.roi_mgr.get_roi(name)
                if info is None:
                    continue
    
                vals = self._current_roi_values_for_mz(
                    info["mask"], mz1, tol1, int(info["z"])
                )
                vals = vals[np.isfinite(vals)]
    
                if vals.size >= 10:
                    data.append(vals)
                    labels.append(name)
    
            if len(data) < 3:
                raise ValueError("Not enough valid ROI data.")
    
            # 🔬 GLOBAL TEST
            res = self._multi_roi_group_test(data)
    
            # significance stars
            if res["p"] < 0.001:
                star = "***"
            elif res["p"] < 0.01:
                star = "**"
            elif res["p"] < 0.05:
                star = "*"
            else:
                star = "n.s."
    
            msg = (
                f"Global test: {res['test']}\n\n"
                f"p-value: {res['p']:.3e} {star}\n\n"
                f"Normality: {'OK' if res['normal'] else 'Not satisfied'}\n"
                f"Variance homogeneity: {'OK' if res['homogenous'] else 'Not satisfied'}"
            )
    
            QMessageBox.information(
                self,
                "Multi-ROI Statistics Result",
                msg
            )
    
            if res["p"] < 0.05:
                # Plot 4’ü aç
                self.plotTabs.setTabEnabled(2, True)
                self.leftTabs.setCurrentIndex(2)
                self.plotTabs.setCurrentIndex(2)
                # Çizimi yap
                pairs = self._pairwise_mwu_bh(data, labels)
                self._plot_posthoc_pairwise_violin(data, labels, pairs)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Multi-ROI Statistics Error",
                str(e)
            )
            
    def _draw_bracket(self, ax, x1, x2, y, h, text, color="#111827"):
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1.4,
            c=color
        )
        ax.text(
            (x1 + x2) * 0.5,
            y + h * 1.05,
            text,
            ha="center",
            va="bottom",
            fontsize = self.PLOT_FONT["title"],
            color=color,
            fontweight="bold"
        )

           
            
            
            
    def _delta_to_color(self, delta):
        d = abs(delta)
        if d < 0.147:
            return "#e5e7eb"   # negligible
        elif d < 0.33:
            return "#bfdbfe"   # small
        elif d < 0.474:
            return "#60a5fa"   # medium
        else:
            return "#1d4ed8"   # large



    def _heavy_tail_warning(self, vals, thresh=0.02):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 20:
            return False
    
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return False
    
        outlier_ratio = np.mean(vals > (q3 + 3 * iqr))
        return outlier_ratio > thresh





    def _plot_posthoc_pairwise_violin(self, data, labels, pairs):

        self.leftTabs.setCurrentIndex(2)
        self.plotTabs.setTabEnabled(2, True)
        self.plotTabs.setCurrentIndex(2)
        QApplication.processEvents()
    
        fig = self.posthocPlot.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
    
        positions = []
        plot_data = []
        plot_labels = []
    
        x = 1.0
        step = 2.6
        pair_centers = []
    
        for p in pairs:
            i, j = p["i"], p["j"]
    
            plot_data.append(data[i])
            plot_data.append(data[j])
    

            plot_labels.extend([
                f"{labels[i]}\n(n={len(data[i]):,})",
                f"{labels[j]}\n(n={len(data[j]):,})"
            ])

            positions.extend([x, x + 1])
    
            pair_centers.append((x + 0.5, p))
            x += step
    
        parts = ax.violinplot(
            plot_data,
            positions=positions,
            widths=0.85,
            showmedians=False,
            showextrema=False
        )
    
        k = 0
        for p in pairs:
            delta = p.get("delta", 0.0)
            color = self._delta_to_color(delta)
    
            for _ in range(2):
                body = parts["bodies"][k]
                body.set_facecolor(color)
                body.set_edgecolor("#1e3a8a")
                body.set_alpha(0.85)
                k += 1
    
        # ---------- MEDIAN + IQR (hover enabled) ----------
        med_scatters = []
    
        for pos, vals in zip(positions, plot_data):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([pos, pos], [q1, q3], lw=4, color="#020617")
    
            sc = ax.scatter(
                pos, med,
                s=30,
                color="#020617",
                zorder=3,
                picker=5
            )
            sc._median_value = med
            med_scatters.append(sc)
    
        # ---------- Y LIMITS ----------
        ymax = max(np.max(v) for v in plot_data if len(v) > 0)
        h = ymax * 0.035
        y_base = ymax * 1.05
    
        # ---------- SIGNIFICANCE BRACKETS ----------
        for center, p in pair_centers:
            star = p.get("star", "n.s.")
            if star == "n.s.":
                continue
    
            delta = p.get("delta", 0.0)
            cat = self._delta_category(delta)
            color = self.DELTA_COLORS.get(cat, "#374151")
    
            x1, x2 = center - 0.5, center + 0.5
    
            self._draw_bracket(
                ax,
                x1=x1,
                x2=x2,
                y=y_base,
                h=h,
                text=star,
                color=color
            )
    
            ax.text(
                (x1 + x2) / 2,
                y_base - h * 0.6,
                f"δ = {delta:.2f}",
                ha="center",
                va="top",
                fontsize=5,
                color=color,
                style="italic"
            )
    
            y_base += h * 1.9
    
        # ---------- OUTLIER AWARENESS ----------
        show_warning = False
        for vals in plot_data:
            if self._heavy_tail_warning(vals):
                show_warning = True
                break
    
        if show_warning:
            ax.text(
                0.5, -0.20,
                "⚠ Heavy-tailed intensity distribution detected",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color="#6b7280",
                style="italic"
            )
    
        # ---------- AXES ----------
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=4)
    
        self._apply_axes_style(
            ax,
            title=(
                "Pairwise post-hoc comparison\n"
                "(Mann–Whitney U, BH-FDR corrected)"),
            fontsize=self.PLOT_FONT["subtitle"],
            ylabel="Intensity"
        )
    
        ax.set_ylim(top=y_base + h)
    
        # ---------- MEDIAN HOVER ----------
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="#9ca3af", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#6b7280")
        )
        annot.set_visible(False)
    
        def on_pick(event):
            artist = event.artist
            if hasattr(artist, "_median_value"):
                x, y = artist.get_offsets()[0]
                annot.xy = (x, y)
                annot.set_text(f"median = {artist._median_value:.3g}")
                annot.set_visible(True)
                self.posthocPlot.canvas.draw_idle()
    
        def on_motion(event):
            if annot.get_visible():
                annot.set_visible(False)
                self.posthocPlot.canvas.draw_idle()
    
        self.posthocPlot.canvas.mpl_connect("pick_event", on_pick)
        self.posthocPlot.canvas.mpl_connect("motion_notify_event", on_motion)
    
        self.posthocPlot.canvas.draw_idle()
    
        
    
            
    def roi_compare_boxplot(self):
        if not self._check_loaded():
            return
    
        nameA = self.comboROI_A.currentText().strip()
        nameB = self.comboROI_B.currentText().strip()
    
        if not nameA or not nameB or nameA == nameB:
            QMessageBox.warning(self, "ROI Compare", "Select two different ROIs.")
            return
    
        mz1, tol1, _, _, _, _, _ = self._read_mz_block()
    
        try:
            self.statusbar.showMessage("📦 ROI Compare (Boxplot + KDE + Stats)...")
    
            data = []    
            for name in (nameA, nameB):
                info = self.roi_mgr.get_roi(name)
                if info is None:
                    continue
    
                vals = self._current_roi_values_for_mz(
                    info["mask"], mz1, tol1, int(info["z"])
                )
                vals = vals[np.isfinite(vals)]
                if vals.size >= 10:
                    data.append(vals)
    
            if len(data) != 2:
                raise ValueError("Not enough valid ROI data.")
    
            valsA, valsB = data
            labels = [
                f"{nameA}\n(n={len(valsA):,})",
                f"{nameB}\n(n={len(valsB):,})"
            ]

            # ================= FIGURE =================
            fig = self.boxPlot.canvas.fig
            fig.clear()
            fig.set_constrained_layout(True)
    
            gs = fig.add_gridspec(
                1, 2,
                width_ratios=[1.15, 1.0],
                wspace=0.28
            )
    
            ax_box = fig.add_subplot(gs[0, 0])
            ax_kde = fig.add_subplot(gs[0, 1])
    
            # ================= BOX PLOT =================
            box = ax_box.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                widths=0.55
            )
    
            for b in box["boxes"]:
                b.set_facecolor("#93c5fd")
                b.set_edgecolor("#1d4ed8")
                b.set_linewidth(1.2)
    
            for m in box["medians"]:
                m.set_color("#020617")
                m.set_linewidth(2.2)
    
            ax_box.set_xticks([1, 2])
            ax_box.set_xticklabels(labels, fontsize=self.PLOT_FONT["tick"])
    
            # axis style (tek tip font)
            self._apply_axes_style(
                ax_box,
                title=f"{nameA} vs {nameB}(Boxplot)",
                ylabel="Intensity",
                fontsize=self.PLOT_FONT["label"]
            )
    
            ax_box.tick_params(
                axis="both",
                labelsize=self.PLOT_FONT["tick"]
            )
    
            # ---- m/z subtitle ----
            self._add_mz_subtitle(
                ax_box, f"m/z {mz1:.4f} ± {tol1}"
            )
    
            # ---- INTENSITY LIMIT (STABLE) ----
            ymin, ymax = ax_box.get_ylim()
            ax_box.set_ylim(
                bottom=max(0, ymin * 0.8),
                top=ymax * 1.15
            )
    
            ax_box.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-2, 2)
            )
    
            # ================= STATISTICS =================
            u_stat, p_val = mannwhitneyu(valsA, valsB, alternative="two-sided")
            delta = self._cliffs_delta(valsA, valsB)
    
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = "n.s."
    
            stat_text = (
                f"{star}\n"
                f"p = {p_val:.2e}\n"
                f"Cliff’s δ = {delta:.3f}"
            )
    
            ax_box.text(
                0.02, 0.02,
                stat_text,
                transform=ax_box.transAxes,
                ha="left",
                va="bottom",
                fontsize=self.PLOT_FONT["stat"],
                bbox=dict(
                    facecolor="#f9fafb",
                    alpha=0.85,
                    edgecolor="#94a3b8",
                    linewidth=0.8,
                    boxstyle="round,pad=0.3"
                )
            )
            annot = ax_box.annotate(
                "",
                xy=(0, 0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="#9ca3af", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#6b7280")
            )
            annot.set_visible(False)
    
            median_lines = box["medians"]
    
            def _on_move_boxplot(event):
                if event.inaxes != ax_box:
                    return
    
                for line in median_lines:
                    contains, _ = line.contains(event)
                    if contains:
                        x = np.mean(line.get_xdata())
                        y = line.get_ydata()[0]
                        annot.xy = (x, y)
                        annot.set_text(f"median = {y:.3g}")
                        annot.set_visible(True)
                        self.boxPlot.canvas.draw_idle()
                        return
    
                if annot.get_visible():
                    annot.set_visible(False)
                    self.boxPlot.canvas.draw_idle()
    
            self.boxPlot.canvas.mpl_connect(
                "motion_notify_event", _on_move_boxplot
            )
    
            # ================= KDE + GAUSSIAN =================
            self._draw_kde_gaussian(ax_kde, data, labels=labels)
    
            ax_kde.set_title(
                "Distribution (KDE, area = 1)",
                fontsize=self.PLOT_FONT["title"],
                fontweight="bold"
            )
            ax_kde.tick_params(axis="both", labelsize=self.PLOT_FONT["tick"])
    
            # ================= FINAL =================
            fig.subplots_adjust(
                top=0.90,
                bottom=0.14,
                left=0.10,
                right=0.97
            )
    
            self.boxPlot.canvas.draw_idle()
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(0)
            self.statusbar.showMessage("✅ ROI Compare (Boxplot + KDE + Stats) ready")
    
        except Exception as e:
            QMessageBox.critical(self, "ROI Compare Error", str(e))    
    
    def roi_compare_violin(self):
        if not self._check_loaded():
            return
    
        nameA = self.comboROI_A.currentText().strip()
        nameB = self.comboROI_B.currentText().strip()
    
        if not nameA or not nameB or nameA == nameB:
            QMessageBox.warning(self, "ROI Compare", "Select two different ROIs.")
            return
    
        mz1, tol1, _, _, _, _, _ = self._read_mz_block()
    
        try:
            self.statusbar.showMessage("📦 ROI Compare (Violin + Stats)...")
    
            data = []
            ns = []
            for name in (nameA, nameB):
                info = self.roi_mgr.get_roi(name)
                if info is None:
                    continue
    
                vals = self._current_roi_values_for_mz(
                    info["mask"], mz1, tol1, int(info["z"])
                )
                vals = vals[np.isfinite(vals)]
                if vals.size >= 10:
                    data.append(vals)
                    ns.append(vals.size)
    
            if len(data) != 2:
                raise ValueError("Not enough valid ROI data.")
    
            valsA, valsB = data
            labels = [
                f"{nameA}\n(n={ns[0]:,})",
                f"{nameB}\n(n={ns[1]:,})"
            ]
    
            # ================= FIGURE =================
            fig = self.multiPlot.canvas.fig
            fig.clear()
            fig.set_constrained_layout(True)
            ax = fig.add_subplot(111)
    
            # ================= VIOLIN =================
            parts = ax.violinplot(
                data,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.75
            )
    
            for pc in parts["bodies"]:
                pc.set_facecolor("#93c5fd")
                pc.set_edgecolor("#2563eb")
                pc.set_alpha(0.85)
                
           # ================= BONCUK (PIXEL-LEVEL SCATTER) =================
            for i, vals in enumerate(data, start=1):
                x = np.random.normal(i, 0.05, size=len(vals))  # jitter
                ax.scatter(
                    x,
                    vals,
                    s=16,
                    color="#020617",
                    alpha=0.35,
                    zorder=2
                )
     
                
            # ================= IQR + MEDIAN OVERLAY =================
            median_points = []
            for i, vals in enumerate(data, start=1):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
    
                # IQR
                ax.plot([i, i], [q1, q3], color="#020617", lw=4, solid_capstyle="round")
                # median
                ax.scatter(i, med, color="#020617", s=30, zorder=3)
                median_points.append((i, med))
    
            # ================= MEDIAN HOVER =================
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                bbox=dict(
                    boxstyle="round",
                    fc="white",
                    ec="#9ca3af",
                    alpha=0.9
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#6b7280"
                )
            )
            annot.set_visible(False)
    
            def _on_move(event):
                if event.inaxes != ax:
                    return
    
                for x, y in median_points:
                    px, py = ax.transData.transform((x, y))
                    dist = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
    
                    if dist < 10:
                        annot.xy = (x, y)
                        annot.set_text(f"median = {y:.3g}")
                        annot.set_visible(True)
                        self.multiPlot.canvas.draw_idle()
                        return
    
                if annot.get_visible():
                    annot.set_visible(False)
                    self.multiPlot.canvas.draw_idle()
    
            self.multiPlot.canvas.mpl_connect("motion_notify_event", _on_move)
    
            # ================= AXES =================
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels, fontsize=self.PLOT_FONT["tick"])
            ax.tick_params(
                    axis="x",
                    pad=6   
                )
    
            self._apply_axes_style(
                ax,
                title=f"{nameA} vs {nameB}(Violin)",
                ylabel="Intensity"
            )
    
            self._add_mz_subtitle(ax, f"m/z {mz1:.4f} ± {tol1}")
    
            # ================= STATISTICS =================
            u_stat, p_val = mannwhitneyu(valsA, valsB, alternative="two-sided")
            delta = self._cliffs_delta(valsA, valsB)
    
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = "n.s."
    
            y_max = max(np.max(valsA), np.max(valsB))
            y = y_max * 1.08
    
            ax.plot([1, 1, 2, 2], [y * 0.98, y, y, y * 0.98],
                    lw=1.2, color="#020617")
            ax.text(
                1.5, y * 1.02,
                star,
                ha="center",
                va="bottom",
                fontsize=self.PLOT_FONT["stat"],
                fontweight="bold"
            )
    
            ax.text(
                0.5, 0.905,   # 🔑 m/z’nin ALTINA
                (
                    "Mann–Whitney U (two-sided)\n"
                    f"p = {p_val:.2e}\n"
                    f"Cliff’s δ = {delta:.3f}"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=self.PLOT_FONT["stat"],
                color="#374151",   # koyu gri
                bbox=dict(
                    facecolor="#f9fafb",
                    alpha=0.85,
                    edgecolor="#d1d5db",
                    linewidth=0.8,
                    boxstyle="round,pad=0.25"
                )
            )

    
            # ================= FINAL =================
            self.multiPlot.canvas.draw_idle()
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(1)
            self.statusbar.showMessage("✅ ROI Compare (Violin + Stats) ready")
    
        except Exception as e:
            QMessageBox.critical(self, "ROI Compare (Violin) Error", str(e))
    
    def _roi_volcano_threaded(self):
        print(">>> VOLCANO CLICKED")
    
        if not self._check_loaded():
            return
    
        if len(self.roi_mgr.saved_rois) < 2:
            QMessageBox.warning(self, "Volcano Plot", "At least two saved ROIs are required.")
            return
    
        nameA = self.comboROI_A.currentText().strip()
        nameB = self.comboROI_B.currentText().strip()
    
        if not nameA or not nameB or nameA == nameB:
            QMessageBox.warning(self, "Volcano Plot", "Please select two different ROIs.")
            return
    
        try:
            tol = float(self.width_textbox.text())
        except Exception:
            tol = 0.5
    
        self.statusbar.showMessage("⏳ Computing ROI Volcano...")
    
        worker = CancelableTaskWorker(
            task_roi_volcano,
            self.parser,
            self.roi_mgr.saved_rois,
            nameA,
            nameB,
            tol
        )
    
        # ================= RESULT =================
        def _on_result(payload):
                
            print("VOLCANO PAYLOAD KEYS:", payload.keys())
            print("len(mz):", len(payload.get("mz", [])))
            print("len(logFC):", len(payload.get("logFC", [])))
            print("len(p_raw):", len(payload.get("p_raw", [])))
            print("reason:", payload.get("reason"))
            
            if payload is None or payload.get("reason"):
                reason = payload.get("reason", "unknown")
                QMessageBox.warning(
                    self,
                    "Volcano Plot",
                    f"No valid volcano data ({reason})."
                )
                self.statusbar.showMessage("⚠️ Volcano not generated")
                return
    
            logfc = payload["logFC"]
            qvals = payload["q"]
    
            if logfc.size == 0:
                QMessageBox.warning(self, "Volcano Plot", "No valid volcano points.")
                return
    
            ax = self.volcanoPlot.canvas.ax
            ax.clear()
    
            y = -np.log10(qvals)
    
            # significance mask
            sig = (qvals < 0.05) & (np.abs(logfc) > 0.5)
    
            # --- background points ---
            ax.scatter(
                logfc[~sig],
                y[~sig],
                s=12,
                alpha=0.35,
                color="#94a3b8",
                label="Not significant"
            )
    
            # --- significant points ---
            ax.scatter(
                logfc[sig],
                y[sig],
                s=18,
                alpha=0.85,
                color="#dc2626",
                label="q < 0.05 & |log2FC| > 0.5"
            )
    
            # --- thresholds ---
            ax.axhline(-np.log10(0.05), ls="--", lw=1, color="#6b7280")
            ax.axvline(0.5, ls="--", lw=1, color="#6b7280")
            ax.axvline(-0.5, ls="--", lw=1, color="#6b7280")
    
            self._apply_axes_style(
                ax,
                title=f"ROI Volcano: {nameA} vs {nameB}",
                xlabel="log2 Fold Change",
                ylabel="-log10(q-value)"
            )
    
            ax.legend(fontsize=7, frameon=False)
    
            self.volcanoPlot.canvas.draw_idle()
            self.plotTabs.setTabEnabled(3, True)            
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(3)
            self.statusbar.showMessage("✅ Volcano plot ready")
    
        # ================= ERROR =================
        def _on_error(msg):
            QMessageBox.critical(self, "Volcano Error", msg)
            self.statusbar.showMessage("❌ Volcano failed")
    
        worker.result.connect(_on_result)
        worker.error.connect(_on_error)
        self.worker_mgr.start(worker)
                

    
# --------------------------- Style helpers ---------------------------
    
    def _apply_axes_style(
        self,
        ax,
        title=None,
        xlabel=None,
        ylabel=None,
        fontsize=None
    ):
        """
        Apply unified axes styling (publication-ready).
        """
    
        if fontsize is None:
            fontsize = self.PLOT_FONT["label"]
    
        if title:
            ax.set_title(
                title,
                fontsize=self.PLOT_FONT["title"],
                fontweight="bold",
                pad=8
            )
    
        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=fontsize
            )
    
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=fontsize
            )
    
        ax.tick_params(
            axis="both",
            labelsize=self.PLOT_FONT["tick"]
        )
    
        ax.grid(True, linestyle="--", alpha=0.35)
    
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    def _on_load_done(self, payload):

        
        parser, info = payload
        self.parser = parser
    
        dataset_id = os.path.basename(self.imzml_filename)
    
        self.datasets[dataset_id] = {
            "parser": parser,
            "global_spectrum": None
        }
        self.active_dataset_id = dataset_id
    
        # -----------------------------
        # Dataset size & mode detection
        # -----------------------------
        total_mb = getattr(self, "last_total_size_mb", 0)
    
        if total_mb > 5000:   # ~5 GB
            self.large_dataset_mode = True
            logger.info("Large dataset mode ENABLED")
            self.statusbar.showMessage(
                f"✅ Loaded (SAFE) | Large dataset mode | "
                f"m/z {info['mz_min']:.2f}–{info['mz_max']:.2f}"
            )
        else:
            self.large_dataset_mode = False
            logger.info("Large dataset mode DISABLED")
            self.statusbar.showMessage(
                f"✅ Loaded | m/z {info['mz_min']:.2f}–{info['mz_max']:.2f}"
            )
    
        # 🔑 CRITICAL: propagate mode to parser (workers see this!)
        parser.large_dataset_mode = self.large_dataset_mode
    
        # -----------------------------
        # Human-readable dataset info
        # -----------------------------
        total_bytes = int(total_mb * 1024**2)
        size_str = self._format_bytes(total_bytes)
    
        pixels = len(parser.coordinates)
        z_planes = (
            len(set(z for (_, _, z) in parser.coordinates))
            if parser.coordinates else 0
        )
    
        mode = "Large Dataset" if self.large_dataset_mode else "Standard"
    
        print("\n📦 Dataset Info")
        print("-" * 30)
        print(f"File: {os.path.basename(self.imzml_filename)}")
        print(f"Pixels: {pixels:,}")
        print(f"Z-planes: {z_planes}")
        print(f"m/z range: {info['mz_min']:.2f} – {info['mz_max']:.2f}")
        print(f"Size: {size_str}")
        print(f"Mode: {mode}\n")
    
        # -----------------------------
        # Benchmark logging
        # -----------------------------
        from datetime import datetime
    
        self.benchmark_log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": os.path.basename(self.imzml_filename),
            "pixels": pixels,
            "size_mb": round(self.last_total_size_mb, 2),
            "load_s": round(self.last_load_time_s or 0.0, 3),
            "ram_mb": round(self.last_peak_ram_mb or 0.0, 1),
            "mode": mode,
        })
    
        # -----------------------------
        # UI state updates
        # -----------------------------
        self._update_plot_button_state()
        self._set_ai_assist_enabled(True)
    




    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open imzML",
            os.getcwd(),
            "imzML Files (*.imzML *.imzml);;All Files (*)"
        )
        if not filename:
            return
    
        self._reset_for_new_file()
    
        try:
            self.statusbar.showMessage("⏳ Loading file (safe)...")
            QApplication.processEvents()
    
            # ---- locate ibd ----
            ibd = filename.replace(".imzML", ".ibd").replace(".imzml", ".ibd")
            if not os.path.exists(ibd):
                raise FileNotFoundError(f".ibd file not found:\n{ibd}")
    
            # ---- total size (MB) ----
            imzml_bytes = os.path.getsize(filename)
            ibd_bytes = os.path.getsize(ibd)
            total_mb = (imzml_bytes + ibd_bytes) / (1024 ** 2)
            self.last_total_size_mb = total_mb
    
            # ---- START SAFE LOAD (THREAD) ----
            self.statusbar.showMessage("⏳ Loading file (safe)...")
            
            self._load_thread = LoadImzMLThread(filename)
            self._load_thread.result.connect(self._on_load_done)
            self._load_thread.error.connect(
                lambda e: QMessageBox.critical(self, "❌ Error", e)
            )
            self._load_thread.start()

    
            # ---- UI-only updates (SAFE) ----
            self.imzml_filename = filename
            self.viewFileName.setText(os.path.basename(filename))
    
            self._ion_cache.clear()
            self._ion_cache_order.clear()
            self.clear_roi()
            self._clear_feature_cache()
    
            self.roi_mgr.clear()
            self.roiList.clear()
            self.comboROI_A.clear()
            self.comboROI_B.clear()
            self._roi_counter = 1
            self._reset_for_new_file()

    
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", f"Failed to load:\n{str(e)}")
            self.statusbar.showMessage("❌ Error loading file")
            self.parser = None
            self.imzml_filename = None
            self._set_ai_assist_enabled(False)
            self._update_plot_button_state()
    
    def _check_loaded(self):
        if not self.parser:
            QMessageBox.warning(self, "⚠️ No File", "Please load an imzML file first")
            return False
        return True

# --------------------------- Ion image cache ---------------------------   
    def _get_ion_image_cached(self, mz, tol, z):
        print("DEBUG _get_ion_image_cached:", mz, tol, z)
        if not hasattr(self, "_ion_image_cache"):
            self._ion_image_cache = {}
            self._tic_cache = {}
        key = (mz, tol, z)
        if key in self._ion_image_cache:
            return self._ion_image_cache[key]
    
        # --- raw ion image ---
        ion_img = getionimage(
            self.parser,
            float(mz),
            float(tol),
            int(z),
            reduce_func=sum)

        self._ion_image_cache[key] = ion_img
        return ion_img
  
    def _detect_msi_modality(self):
        """
        Heuristic MSI modality detection from filename + metadata.
        Returns: 'MALDI', 'AP-SMALDI', 'DESI', 'LA-ESI', 'LTP', or 'UNKNOWN'
        """
        name = (self.imzml_filename or "").lower()
    
        # --- filename heuristics ---
        if "laesi" in name or "la-esi" in name:
            return "LA-ESI"
        if "desi" in name:
            return "DESI"
        if "ltp" in name or "plasma" in name:
            return "LTP"
        if "ap-smaldi" in name or "apsmaldi" in name:
            return "AP-SMALDI"
        if "maldi" in name:
            return "MALDI"
    
        # --- fallback ---
        return "UNKNOWN"


    def _select_global_spectrum_stat(self, modality):
        """
        Select aggregation strategy based on MSI modality.
        """
        if modality == "MALDI":
            return "median"
        if modality == "AP-SMALDI":
            return "mean"        
        if modality == "DESI":
            return "mean"
        if modality == "LTP":
            return "sum"
        if modality == "LA-ESI":
            return "median"
        return "mean"
    
        
        
    def _get_global_spectrum(self):
        if not self._check_loaded():
            return None, None
    
        ds = self.active_dataset_id
        if ds in self._global_spectrum_cache:
            return self._global_spectrum_cache[ds]
    
        modality = self._detect_msi_modality()
        stat = self._select_global_spectrum_stat(modality)
    
        max_pixels = 600 if self.large_dataset_mode else 3000
    
        mz, inten = compute_global_spectrum_binned(
            self.parser,
            max_pixels=max_pixels,
            n_bins=6000,
            stat=stat
        )
    
        mz_min = getattr(self, "current_mz_min", 50.0)
        mz_max = getattr(self, "current_mz_max", 5000.0)
    
        mask = (
            np.isfinite(mz) &
            np.isfinite(inten) &
            (mz >= mz_min) &
            (mz <= mz_max) &
            (inten >= 0)
        )
    
        mz = mz[mask]
        inten = inten[mask]
    
        self._global_spectrum_cache[ds] = (mz, inten)
        return mz, inten
    
    

    def process_spectrum(self):
        if not self._check_loaded():
            return
    
        try:
            df_mz, df_int = self._get_global_spectrum()
            if df_mz is None:
                return
    
            ax = self.spectrumPlot.canvas.ax
            ax.clear()
            ax.set_axis_on()
            ax.plot(df_mz, df_int, linewidth=1.2)
    
            # --- CLEAR OLD PEAK MARKERS ---
            if hasattr(self, "_scatter_top_peaks") and self._scatter_top_peaks:
                try:
                    self._scatter_top_peaks.remove()
                except Exception:
                    pass
                self._scatter_top_peaks = None
    
            # --- TOP PEAK DETECTION ---
            if hasattr(self, "chkTopPeaks") and self.chkTopPeaks.isChecked():
                try:
                    from scipy.signal import find_peaks
                    N = 5
                    peaks, props = find_peaks(
                        df_int,
                        distance=20,
                        prominence=np.max(df_int) * 0.01
                    )
    
                    if len(peaks) > 0:
                        top_idx = np.argsort(df_int[peaks])[-N:]
                        self._top_peaks_idx = peaks[top_idx]
                        self._top_peaks_mz = df_mz[self._top_peaks_idx]
                        self._top_peaks_int = df_int[self._top_peaks_idx]
    
                        self._scatter_top_peaks = ax.scatter(
                            self._top_peaks_mz,
                            self._top_peaks_int,
                            color="red",
                            s=40,
                            zorder=5,
                            label=f"Top {len(self._top_peaks_mz)} peaks",
                            picker=True,
                            pickradius=5
                        )
    
                        for mzv in self._top_peaks_mz:
                            ax.axvline(
                                mzv,
                                color="red",
                                alpha=0.25,
                                linestyle="--"
                            )
                        ax.legend()
                    else:
                        self._top_peaks_idx = []
                        self._top_peaks_mz = []
                        self._top_peaks_int = []
    
                except Exception:
                    self.statusbar.showMessage("⚠️ Peak detection skipped (SciPy not available)")
            else:
                self._top_peaks_idx = []
                self._top_peaks_mz = []
                self._top_peaks_int = []
    
            # --- STYLING ---
            self._apply_axes_style(
                ax,
                title="Global Spectrum",
                xlabel="m/z",
                ylabel="Intensity"
            )
    
            self.spectrumPlot.canvas.draw_idle()
            self.leftTabs.setCurrentIndex(0)
            self.aiAssistBtn.setEnabled(True)
            self.statusbar.showMessage("✅ Spectrum processed")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
            self.statusbar.showMessage("❌ Error processing spectrum")
            self._set_ai_assist_enabled(True)
    


    def generate_roi_from_top_peaks(self):
        if not self._check_loaded():
            return        
    
        if (not hasattr(self, "_top_peaks_mz") 
            or self._top_peaks_mz is None 
            or len(self._top_peaks_mz) == 0):
            QMessageBox.information(self,"Top Peaks → ROI",
                                    "No top peaks found. First enable 'Detect Top Peaks' and process spectrum.")
            return
        max_n = len(self._top_peaks_mz)
        default_n = min(3, max_n)
    
        n, ok = QInputDialog.getInt(self,"Top Peaks → ROI",
                                    "How many top peaks to convert into ROI?",
                                    value=default_n,min=1,max=max_n)
        if not ok:
            return    
        try:
            self.statusbar.showMessage("🎯 Generating ROI from top peaks...")

            try:
                tol = validate_numeric_input(self.width_textbox.text(), "tol")
            except Exception:
                tol = 0.5
    
            try:
                z = validate_numeric_input(self.z_textbox.text(), "z", allow_zero=True)
            except Exception:
                z = 1
    
            for i, mz in enumerate(self._top_peaks_mz[:n], start=1):
                img = self._get_ion_image_cached(float(mz), tol, z)
    
                nonzero = img[img > 0]
                if nonzero.size == 0:
                    continue
    
                threshold = np.percentile(nonzero, 95)
                mask = img >= threshold    
                roi_name = f"TopPeak_{i}_mz{mz:.4f}"
    
                self.roi_mgr.add_roi(name=roi_name,mask=mask,z=z)
                self.roiList.addItem(roi_name)
                self.comboROI_A.addItem(roi_name)
                self.comboROI_B.addItem(roi_name)
        
            self.statusbar.showMessage("✅ ROI generated from top peaks")
            
        except Exception as e:
            QMessageBox.critical(self, "ROI Error", str(e))
            self.statusbar.showMessage("❌ ROI generation failed")

    def _set_ai_assist_enabled(self, enabled: bool):
        self.aiAssistBtn.setEnabled(enabled)
        self.aiAction.setEnabled(enabled)       
        
    def ai_assist_top_peak_image(self):
        if not self._check_loaded():
            return
    
        try:
            # ---- UI feedback ----
            self.statusbar.showMessage("🧠 AI Assist: searching top peak...")
    
            QApplication.processEvents()  # UI donmasın

            max_pixels = 600 if self.large_dataset_mode else 000
            
            df_mz, df_int = self._get_global_spectrum()

    
            if df_mz.size == 0:
                raise ValueError("Empty spectrum")
    
            QApplication.processEvents()

            top_idx = int(np.argmax(df_int))
            top_mz = float(df_mz[top_idx])

            self.targetA_textbox.setText(f"{top_mz:.4f}")
            if not self.width_textbox.text().strip():
                self.width_textbox.setText("0.5")
            if not self.z_textbox.text().strip():
                self.z_textbox.setText("1")
    
            QApplication.processEvents()
    
            # 4️⃣ Image-1 üret
            mz = validate_numeric_input(self.targetA_textbox.text(), "m/z1")
            tol = validate_numeric_input(self.width_textbox.text(), "tol")
            z = validate_numeric_input(self.z_textbox.text(), "z", allow_zero=True)
    
            full = self._get_ion_image_cached(mz, tol, z)
            cropped, rix, cix = crop_zero_rows_cols(full)
    
            self._draw_image_on(
                self.image1Plot,
                cropped_img=cropped,
                full_img=full,
                title=f"Image 1 (AI Top Peak): m/z {mz:.4f} ± {tol}",
                row_idx=rix,
                col_idx=cix,
                full_shape=full.shape,
                z_used=z,
                mz=mz,
                tol=tol
            )
    

            self.leftTabs.setCurrentIndex(1)
            self.imageTabs.setCurrentIndex(0)
            self.statusbar.showMessage(
                f"✅ AI Assist → Image 1 generated (m/z {top_mz:.4f})"
            )
    
    
        except Exception as e:
            QMessageBox.critical(self, "AI Assist Error", str(e))
            self.statusbar.showMessage("❌ AI Assist error")





    # --------------------------- m/z reading ---------------------------
    def _read_mz_block(self):
        mz1 = validate_numeric_input(self.targetA_textbox.text(), "m/z1")
        tol1 = validate_numeric_input(self.width_textbox.text(), "tol1")
        z = validate_numeric_input(self.z_textbox.text(), "z", allow_zero=True)
    
        if not self.multiCheck.isChecked():
            return (mz1, tol1, None, None, None, None, z)
    
        mz2_txt = self.targetB_textbox.text().strip()
        tol2_txt = self.width_textboxb.text().strip()
        if not (mz2_txt and tol2_txt):
            return (mz1, tol1, None, None, None, None, z)
    
        mz2 = validate_numeric_input(mz2_txt, "m/z2")
        tol2 = validate_numeric_input(tol2_txt, "tol2")
    
        # mz3 opsiyonel
        mz3_txt = self.targetC_textbox.text().strip()
        tol3_txt = self.width_textboxc.text().strip()
        if mz3_txt and tol3_txt:
            mz3 = validate_numeric_input(mz3_txt, "m/z3")
            tol3 = validate_numeric_input(tol3_txt, "tol3")
        else:
            mz3, tol3 = None, None
    
        return (mz1, tol1, mz2, tol2, mz3, tol3, z)
       
    # --------------------------- Image actions ---------------------------
    def on_image_processing_clicked(self):
        if not self._check_loaded():
            return
        if self.multiCheck.isChecked():
            self.update_images()
        else:
            self.process_image1_only()


    def process_image1_only(self):
        if not self._check_loaded():
            return
        try:
            mz1 = validate_numeric_input(self.targetA_textbox.text(), "m/z1")
            tol1 = validate_numeric_input(self.width_textbox.text(), "tol1")
            z = validate_numeric_input(self.z_textbox.text(), "z", allow_zero=True)
    
            self.statusbar.showMessage(f"⏳ Generating Image 1 (m/z {mz1:.4f})...")
    
            # ==============================
            # BENCHMARK START (render time)
            # ==============================
            t0 = time.perf_counter()
    
            full = self._get_ion_image_cached(mz1, tol1, z)
            cropped, rix, cix = crop_zero_rows_cols(full)
            self._draw_image_on(
                self.image1Plot,
                cropped,
                full,
                f"Image 1: m/z {mz1:.4f} ± {tol1}",
                rix,
                cix,
                full.shape,
                z,
                mz1,
                tol1
            )
    
            # draw queue flush + BENCHMARK END
            QApplication.processEvents()
            t1 = time.perf_counter()
            self.last_image_render_time_s = (t1 - t0)
            # ==============================
            # BENCHMARK END
            # ==============================
    
            self.leftTabs.setCurrentIndex(1)
            self.imageTabs.setCurrentIndex(0)
    
            self.statusbar.showMessage(
                f"✅ Image 1 generated | render={self.last_image_render_time_s:.2f}s"
            )
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
            self.statusbar.showMessage("❌ Error generating image")


    def update_images(self):
        if not self._check_loaded():
            return
        try:
            mz1, tol1, mz2, tol2, mz3, tol3, z = self._read_mz_block()
            self.statusbar.showMessage("⏳ Updating Images...")
#=============== BENCHMARK START (total) ==============================
            t_total0 = time.perf_counter()
            t10 = time.perf_counter()   # ---- Image 1 timing ----
            full1 = self._get_ion_image_cached(mz1, tol1, z)
            img1, r1, c1 = crop_zero_rows_cols(full1)
            self._draw_image_on(
                self.image1Plot, img1, full1,
                f"Image 1: m/z {mz1:.4f} ± {tol1}",
                r1, c1, full1.shape, z, mz1, tol1)
            QApplication.processEvents()
            t11 = time.perf_counter()
            self.last_image1_render_time_s = (t11 - t10)
            self.last_image2_render_time_s = None # ---- Optional: Image 2/3 timing ----
            self.last_image3_render_time_s = None
    
            if self.multiCheck.isChecked() and (mz2 is not None) and (tol2 is not None):
                # Image 2
                t20 = time.perf_counter()
                full2 = self._get_ion_image_cached(mz2, tol2, z)
                img2, r2, c2 = crop_zero_rows_cols(full2)
                self._draw_image_on(
                    self.image2Plot, img2, full2,
                    f"Image 2: m/z {mz2:.4f} ± {tol2}",
                    r2, c2, full2.shape, z, mz2, tol2
                )
                QApplication.processEvents()
                t21 = time.perf_counter()
                self.last_image2_render_time_s = (t21 - t20)
            if (mz3 is not None) and (tol3 is not None):
    
                # Image 3
                t30 = time.perf_counter()
                full3 = self._get_ion_image_cached(mz3, tol3, z)
                img3, r3, c3 = crop_zero_rows_cols(full3)
                self._draw_image_on(
                    self.image3Plot, img3, full3,
                    f"Image 3: m/z {mz3:.4f} ± {tol3}",
                    r3, c3, full3.shape, z, mz3, tol3
                )
                QApplication.processEvents()
                t31 = time.perf_counter()
                self.last_image3_render_time_s = (t31 - t30)
    
            # ---- Total timing end ----
            t_total1 = time.perf_counter()
            self.last_multi_render_total_s = (t_total1 - t_total0)
    
#================= PRINT (background console log) ===========================
            if self.multiCheck.isChecked():
                print(
                    "[BENCH] Render times | "
                    f"Image1={self.last_image1_render_time_s:.3f}s, "
                    f"Image2={self.last_image2_render_time_s:.3f}s, "
                    f"Image3={self.last_image3_render_time_s:.3f}s, "
                    f"TOTAL={self.last_multi_render_total_s:.3f}s"
                )
            else:
                print(
                    "[BENCH] Render times | "
                    f"Image1={self.last_image1_render_time_s:.3f}s, "
                    f"TOTAL={self.last_multi_render_total_s:.3f}s" )
    
            self.leftTabs.setCurrentIndex(1)    
            
            if self.multiCheck.isChecked():
                self.statusbar.showMessage(
                    f"✅ Images updated | I1={self.last_image1_render_time_s:.2f}s "
                    f"I2={self.last_image2_render_time_s:.2f}s "
                    f"I3={self.last_image3_render_time_s:.2f}s "
                    f"TOTAL={self.last_multi_render_total_s:.2f}s"
                )
            else:
                self.statusbar.showMessage(
                    f"✅ Image updated | I1={self.last_image1_render_time_s:.2f}s "
                    f"TOTAL={self.last_multi_render_total_s:.2f}s"
                )
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
            self.statusbar.showMessage("❌ Error updating images")


    def _draw_image_on(self, mpl_widget: MplWidget, cropped_img, full_img, title,
                       row_idx, col_idx, full_shape, z_used, mz, tol):
        fig = mpl_widget.canvas.fig
        ax = mpl_widget.canvas.ax
    
        ax.clear()
        ax.axis("off")
    
        im = ax.imshow(
            cropped_img,
            aspect="auto",
            interpolation=self.cb2.currentText(),
            cmap=self.cb.currentText()
        )
    
        # --- STABLE COLORBAR (single path, fixed cax) ---
        # remove previous cbar
        if getattr(mpl_widget, "cbar", None) is not None:
            try:
                mpl_widget.cbar.remove()
            except Exception:
                pass
            mpl_widget.cbar = None
    
        # remove previous cax axis
        if getattr(mpl_widget, "cax", None) is not None:
            try:
                mpl_widget.cax.remove()
            except Exception:
                pass
            mpl_widget.cax = None
    
        divider = make_axes_locatable(ax)
        mpl_widget.cax = divider.append_axes("right", size="4.5%", pad=0.05)
        mpl_widget.cbar = fig.colorbar(im, cax=mpl_widget.cax)
        # --- Colorbar font control ---
        mpl_widget.cbar.ax.tick_params(labelsize=8)  # sağdaki 0-100 gibi sayılar
        
        mpl_widget.cbar.set_label( "Intensity", fontsize=8,fontweight="bold",labelpad=9)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    
        # store mapping for ROI
        mpl_widget.roi_row_idx = row_idx
        mpl_widget.roi_col_idx = col_idx
        mpl_widget.roi_full_shape = full_shape
        mpl_widget.roi_z = z_used
        
        # 🔑 GLOBAL CROP OFFSET (AU dahil tüm datasetler için)
        if row_idx is not None and len(row_idx) > 0:
            self._crop_row0 = int(row_idx[0])
        else:
            self._crop_row0 = 0
        
        if col_idx is not None and len(col_idx) > 0:
            self._crop_col0 = int(col_idx[0])
        else:
            self._crop_col0 = 0

        mpl_widget.last_full_image = full_img
        mpl_widget.last_mz = mz
        mpl_widget.last_tol = tol

        if mpl_widget.roi_overlay is not None:
            try:
                mpl_widget.roi_overlay.remove()
            except Exception:
                pass
            mpl_widget.roi_overlay = None
    
        if mpl_widget.roi_mask_overlay is not None:
            try:
                mpl_widget.roi_mask_overlay.remove()
            except Exception:
                pass
            mpl_widget.roi_mask_overlay = None
    
        mpl_widget.canvas.draw_idle()

        if hasattr(mpl_widget, "saved_roi_patches"):
            for patch in mpl_widget.saved_roi_patches.values():
                try:
                    ax.add_patch(patch)
                except Exception:
                    pass
        
        # 🏷️ Kalıcı ROI label'ları
        if hasattr(mpl_widget, "roi_labels"):
            for txt in mpl_widget.roi_labels.values():
                try:
                    ax.add_artist(txt)
                except Exception:
                    pass
        mpl_widget.canvas.draw_idle()



    # --------------------------- Plot actions ---------------------------

    def _get_nonzero_flat(self, mz, tol, z):
        img = self._get_ion_image_cached(float(mz), float(tol), int(z))
        v = img.ravel().astype(float)
        v = v[np.isfinite(v)]
    
        # 🔥 0'ları tamamen atma
        if v.size > 0:
            v = v[v > 0]
    
        return v

    
    def _draw_dist_on(self, mpl_widget: MplWidget, data, title, kind="box"):
        ax = mpl_widget.canvas.ax
        ax.clear()
    
        # ------------------ GUARD ------------------
        if data is None:
            ax.text(0.5, 0.5, "No data.", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            mpl_widget.canvas.draw_idle()
            return
    
        # 🔥 TEK ARRAY → LIST OF ARRAY
        if isinstance(data, np.ndarray):
            data = [data]
    
        # remove empty / nan-only groups
        clean = []
        for d in data:
            d = np.asarray(d, dtype=float)
            d = d[np.isfinite(d)]
            if d.size >= 10:
                clean.append(d)
    
        if len(clean) == 0:
            ax.text(
                0.5, 0.5,
                "No valid intensity values\n(check m/z & tolerance)",
                ha="center", va="center", fontsize=11
            )
            ax.set_axisbelow(True)
            mpl_widget.canvas.draw_idle()
            return

    
        # ------------------ PLOT ------------------
        if kind == "violin":
            parts = ax.violinplot(
                clean,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
    
            # 🎨 violin style (box ile uyumlu)
            for pc in parts["bodies"]:
                pc.set_facecolor("#93c5fd")
                pc.set_edgecolor("#2563eb")
                pc.set_alpha(0.85)
    
            if "cmedians" in parts:
                parts["cmedians"].set_color("#111827")
                parts["cmedians"].set_linewidth(2)
    
        else:
            box = ax.boxplot(
                clean,
                vert=True,
                patch_artist=True,
                showfliers=False   # 🔥 violin ile uyum
            )
    
            for b in box["boxes"]:
                b.set_facecolor("#93c5fd")
                b.set_edgecolor("#2563eb")
    
            for m in box["medians"]:
                m.set_color("#111827")
                m.set_linewidth(2)
    
        # ------------------ STYLE ------------------
        self._apply_axes_style(
            ax,
            title=title,
            ylabel="Intensity"
        )
        ax.set_axis_on()
    
        mpl_widget.canvas.draw_idle()
        mpl_widget.canvas.draw_idle()
        mpl_widget.canvas.flush_events()
        mpl_widget.repaint()
        

    def process_box_only(self):
        try:
            self.statusbar.showMessage("⏳ Generating Box Plot(s)...")
            self.update_plots_like_images(kind="box")
            self.statusbar.showMessage("✅ Box plot(s) generated")
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
            self.statusbar.showMessage("❌ Error generating box plot")
    
    def process_violin_only(self):
        try:
            self.statusbar.showMessage("⏳ Generating Violin Plot(s)...")
            self.update_plots_like_images(kind="violin")
            self.statusbar.showMessage("✅ Violin plot(s) generated")
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
            self.statusbar.showMessage("❌ Error generating violin plot")

    # --------------------------- ROI helpers ---------------------------

    def _active_image_widget(self) -> MplWidget:
        idx = self.imageTabs.currentIndex()
        return [self.image1Plot, self.image2Plot, self.image3Plot][idx]

    def _disconnect_roi_selector(self):
        if self.roi_selector is not None:
            try:
                self.roi_selector.set_active(False)
            except Exception:
                pass
        self.roi_selector = None

    def _cliffs_delta(self, x, y):

        x = np.asarray(x)
        y = np.asarray(y)
    
        nx = len(x)
        ny = len(y)
    
        gt = 0
        lt = 0
        for xi in x:
            gt += np.sum(xi > y)
            lt += np.sum(xi < y)
    
        return (gt - lt) / (nx * ny)

    def _draw_kde_gaussian(self, ax, data, labels):    
        colors = ["#2563eb", "#dc2626"]      # ROI 1, ROI 2
        linestyles = ["-", "--"]
    
        ax.clear()
    
        # ortak x ekseni (intensity)
        all_vals = np.concatenate(data)
        x_min, x_max = np.percentile(all_vals, [1, 99])
        x = np.linspace(x_min, x_max, 400)
    
        for i, vals in enumerate(data):
            vals = vals[np.isfinite(vals)]
            if vals.size < 10:
                continue
    
            # ================= ROBUST RANGE (BOXPLOT-ALIGNED) =================
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
    
            lo = q1 - 1.0 * iqr
            hi = q3 + 1.0 * iqr
    
            vals_kde = vals[(vals >= lo) & (vals <= hi)]
            if vals_kde.size < 10:
                vals_kde = vals  # fallback
    
            # ================= ADAPTIVE BANDWIDTH =================
            std = np.std(vals_kde, ddof=1)
            if std > 0:
                bw = max(0.35, min(0.65, iqr / std))
            else:
                bw = 0.5
    
            # ================= KDE =================
            kde = gaussian_kde(vals_kde, bw_method=bw)
            y_kde = kde(x)
    
            # normalize (area = 1)
            area = np.trapz(y_kde, x)
            if area > 0:
                y_kde /= area
    
            ax.plot(
                x, y_kde,
                color=colors[i],
                linewidth=2.6,
                label=f"{labels[i]} (KDE)"
            )
    
            # ================= GAUSSIAN (REFERENCE) =================
            mu = np.mean(vals)
            sigma = np.std(vals, ddof=1)
            if sigma > 0:
                y_gauss = norm.pdf(x, mu, sigma)
                area_g = np.trapz(y_gauss, x)
                if area_g > 0:
                    y_gauss /= area_g
    
                ax.plot(
                    x, y_gauss,
                    color=colors[i],
                    linestyle=linestyles[1],
                    linewidth=2.0,
                    alpha=0.65,
                    label=f"{labels[i]} (Gaussian)"
                )
    
        # ================= AXES STYLE =================
        ax.set_xlabel("Intensity", fontsize=8)
        ax.set_ylabel("Density (area = 1)", fontsize=8)
    
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, frameon=True)
    
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    def _multi_roi_group_test(self, data):
    
        if not _HAS_SCIPY_STATS:
            return {
                "test": "Not computed (SciPy missing)",
                "stat": float("nan"),
                "p": float("nan"),
                "normal": False,
                "homogenous": False
            }
    
        # ---------- CLEAN ----------
        clean = []
        for vals in data:
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size >= 5:
                clean.append(v)
    
        if len(clean) < 3:
            return {
                "test": "Not computed (<3 ROIs)",
                "stat": float("nan"),
                "p": float("nan"),
                "normal": False,
                "homogenous": False
            }
    
        # ---------- NORMALITY (Shapiro) ----------
        normal_flags = []
        for vals in clean:
            if vals.size > 5000:
                vals = np.random.choice(vals, 5000, replace=False)
            try:
                _, p_norm = shapiro(vals)
                normal_flags.append(p_norm > 0.05)
            except Exception:
                normal_flags.append(False)
    
        all_normal = all(normal_flags)
    
        # ---------- HOMOGENEITY ----------
        try:
            _, p_lev = levene(*clean)
            homogenous = p_lev > 0.05
        except Exception:
            homogenous = False
    
        # ---------- DECISION ----------
        if all_normal and homogenous:
            stat, pval = f_oneway(*clean)
            test_name = "One-way ANOVA"
        else:
            stat, pval = kruskal(*clean)
            test_name = "Kruskal–Wallis"
    
        return {
            "test": test_name,
            "stat": float(stat),
            "p": float(pval),
            "normal": all_normal,
            "homogenous": homogenous
        }



    def _pairwise_mwu_bh(self, data, labels, alpha=0.05):
        results = []
        pvals = []
    
        pairs = list(combinations(range(len(data)), 2))
    
        # ---------- PAIRWISE TESTS ----------
        for i, j in pairs:
            u, p = mannwhitneyu(
                data[i], data[j],
                alternative="two-sided"
            )
    
            delta = self._cliffs_delta(data[i], data[j])
    
            pvals.append(p)
    
            results.append({
                "i": i,
                "j": j,
                "label": f"{labels[i]} vs {labels[j]}",
                "p_raw": float(p),
                "delta": float(delta)
            })
    
        # ---------- BENJAMINI–HOCHBERG ----------
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
    
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, m + 1) / m)
    
        passed = ranked <= thresh
        cutoff = ranked[np.where(passed)[0].max()] if passed.any() else 0.0
    
        # ---------- ASSIGN STARS ----------
        for r in results:
            p = r["p_raw"]
            r["p_adj"] = float(p) if p <= cutoff else 1.0
    
            if r["p_adj"] < 0.001:
                r["star"] = "***"
            elif r["p_adj"] < 0.01:
                r["star"] = "**"
            elif r["p_adj"] < 0.05:
                r["star"] = "*"
            else:
                r["star"] = "n.s."
    
        return results
        
    def clear_roi(self):
        if self.parser is None:
            return
    
        self._disconnect_roi_selector()
    
        self.roi_mgr.clear()
        self.roi_mgr.last_roi_mask = None
        self.roi_mgr.last_roi_z = None
        self.roi_dirty = False
    
        self.roiList.clear()
        self.comboROI_A.clear()
        self.comboROI_B.clear()
        self.lblROIst.setText("ROI: none")
        
        # ROI selection state reset
        self.comboROI_A.setCurrentIndex(-1)
        self.comboROI_B.setCurrentIndex(-1)
       
        for w in (self.image1Plot, self.image2Plot, self.image3Plot):
            if getattr(w, "roi_overlay", None):
                w.roi_overlay.remove()
                w.roi_overlay = None
    
            if getattr(w, "roi_mask_overlay", None):
                w.roi_mask_overlay.remove()
                w.roi_mask_overlay = None
    
            if hasattr(w, "roi_labels"):
                for lbl in w.roi_labels.values():
                    lbl.remove()
                w.roi_labels.clear()
    
            if hasattr(w, "saved_roi_patches"):
                for p in w.saved_roi_patches.values():
                    p.remove()
                w.saved_roi_patches.clear()
    
            w.canvas.draw_idle()
    
        self.statusbar.showMessage("🧹 ROI cleared (data preserved)")

        
    def start_roi_rectangle(self):
        if not self._check_loaded():
            return
    
        self.leftTabs.setCurrentIndex(1)
        self._disconnect_roi_selector()
    
        w = self._active_image_widget()
        ax = w.canvas.ax
    
        def onselect(eclick, erelease):
            if w.roi_row_idx is None or w.roi_col_idx is None or w.roi_full_shape is None:
                QMessageBox.warning(self, "ROI", "Please generate an image first.")
                return
    
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or x1 is None or y0 is None or y1 is None:
                return
    
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
    
            xmin_i = int(np.floor(xmin))
            xmax_i = int(np.ceil(xmax))
            ymin_i = int(np.floor(ymin))
            ymax_i = int(np.ceil(ymax))
    
            xmin_i = max(0, xmin_i)
            ymin_i = max(0, ymin_i)
            xmax_i = min(len(w.roi_col_idx) - 1, xmax_i)
            ymax_i = min(len(w.roi_row_idx) - 1, ymax_i)
    
            full_cols = w.roi_col_idx[xmin_i:xmax_i + 1]
            full_rows = w.roi_row_idx[ymin_i:ymax_i + 1]
    
            mask = np.zeros(w.roi_full_shape, dtype=bool)
            mask[np.ix_(full_rows, full_cols)] = True
    
            self.roi_mgr.last_roi_mask = mask
            self.roi_mgr.last_roi_z = int(w.roi_z)
            self.roi_dirty = True
    
            # ---------- Overlay temizle ----------
            if getattr(w, "roi_overlay", None) is not None:
                try:
                    w.roi_overlay.remove()
                except Exception:
                    pass
                w.roi_overlay = None

            # ---------- Rectangle çiz ----------
            rect = Rectangle(
                (xmin_i, ymin_i),
                (xmax_i - xmin_i),
                (ymax_i - ymin_i),
                fill=False,
                linewidth=2.0,
                edgecolor="red"
            )
            w.roi_overlay = ax.add_patch(rect)
    
            # ---------- ROI LABEL ----------
            cx = (xmin_i + xmax_i) / 2
            cy = (ymin_i + ymax_i) / 2
    
    
            w.canvas.draw_idle()
    
            # ---------- Bilgi ----------
            npx = int(mask.sum())
            self.lblROIst.setText(
                f"ROI: Rectangle | pixels={npx} | z={w.roi_z}"
            )
    
        self.roi_selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True
        )
    
        self.statusbar.showMessage("🧩 ROI Rectangle: drag on image to select")

    def start_roi_lasso(self):
        if not self._check_loaded():
            return
    
        self.leftTabs.setCurrentIndex(1)
        self._disconnect_roi_selector()
    
        w = self._active_image_widget()
        ax = w.canvas.ax
    
        def onselect(verts):
            if w.roi_row_idx is None or w.roi_col_idx is None or w.roi_full_shape is None:
                QMessageBox.warning(self, "ROI", "Please generate an image first.")
                return
    
            verts = np.asarray(verts)
            if verts.shape[0] < 3:
                return
    
            ny = len(w.roi_row_idx)
            nx = len(w.roi_col_idx)
    
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            points = np.vstack((x.ravel(), y.ravel())).T
    
            path = Path(verts)
            inside = path.contains_points(points).reshape((ny, nx))
    
            mask = np.zeros(w.roi_full_shape, dtype=bool)
            full_rows = w.roi_row_idx
            full_cols = w.roi_col_idx
    
            rr, cc = np.where(inside)
            if rr.size == 0:
                return
    
            mask[full_rows[rr], full_cols[cc]] = True
    
            self.roi_mgr.last_roi_mask = mask
            self.roi_mgr.last_roi_z = int(w.roi_z)
            self.roi_dirty = True
    
            # ---------- Overlay temizle ----------
            if getattr(w, "roi_overlay", None) is not None:
                try:
                    w.roi_overlay.remove()
                except Exception:
                    pass
                w.roi_overlay = None

            # ---------- Lasso patch ----------
            patch = PathPatch(
                path,
                fill=False,
                linewidth=2.0,
                edgecolor="red"
            )
            w.roi_overlay = ax.add_patch(patch)
            w.canvas.draw_idle()
    
            # ---------- Bilgi ----------
            npx = int(mask.sum())
            self.lblROIst.setText(
                f"ROI: Lasso | pixels={npx} | z={w.roi_z}"
            )
    
        self.roi_selector = LassoSelector(ax, onselect=onselect)
        self.statusbar.showMessage("🧩 ROI Lasso: draw polygon on image")


    def roi_mean_spectrum(self):
        if not self._check_loaded():
            return
        if self.roi_mgr.last_roi_mask is None:           
            QMessageBox.warning(self, "ROI Spectrum", "Please select an ROI first (Rectangle or Lasso).")
            return
        z = self.roi_mgr.last_roi_z
        if z is None:
            QMessageBox.warning(
                self,
                "ROI Spectrum",
                "ROI is not linked to a z-plane. Generate image and reselect ROI."
            )
            return


        mask = self.roi_mgr.last_roi_mask
        z = int(self.roi_mgr.last_roi_z)


        try:
            self.statusbar.showMessage("⏳ ROI Mean Spectrum...")

            mzs_list, int_list = [], []
            for i, (x, y, z_) in enumerate(self.parser.coordinates):
                if z_ != z:
                    continue
                if mask[y - 1, x - 1]:
                    mzs, ints = self.parser.getspectrum(i)
                    if len(mzs) > 0:
                        mzs_list.append(np.asarray(mzs))
                        int_list.append(np.asarray(ints))

            if not mzs_list:
                raise ValueError("No spectra found inside ROI (check z and ROI location).")

            lengths = [len(m) for m in mzs_list]
            if len(set(lengths)) == 1:
                df_mz = np.mean(np.vstack(mzs_list), axis=0)
                df_int = np.mean(np.vstack(int_list), axis=0)
            else:
                all_mz = np.concatenate(mzs_list)
                all_int = np.concatenate(int_list)
                bins = np.linspace(all_mz.min(), all_mz.max(), 6000)
                dig = np.digitize(all_mz, bins)
                df_mz, df_int = [], []
                for j in range(1, len(bins)):
                    m = (dig == j)
                    if m.any():
                        df_mz.append(bins[j])
                        df_int.append(all_int[m].mean())
                df_mz = np.array(df_mz)
                df_int = np.array(df_int)

            ax = self.spectrumPlot.canvas.ax
            ax.clear()
            ax.plot(df_mz, df_int, linewidth=1.2)
            self._apply_axes_style(ax, title=f"ROI Mean Spectrum (z={z})", xlabel="m/z", ylabel="Intensity")
            self.spectrumPlot.canvas.draw()

            self.leftTabs.setCurrentIndex(0)
            self.statusbar.showMessage("✅ ROI spectrum ready")
        except Exception as e:
            QMessageBox.critical(self, "❌ ROI Spectrum Error", str(e))
            self.statusbar.showMessage("❌ ROI spectrum error")
    # --------------------------- ROI Analysis (Stats / Save / Overlay / Corr / Compare / CSV) ---------------------------
    def _current_roi_values_for_mz(self, roi_mask, mz, tol, z):
        if mz is None:
            vals = np.ones(int(np.sum(roi_mask)), dtype=float)
            return vals
        img = self._get_ion_image_cached(float(mz),float(tol),int(z))
        return img[roi_mask].astype(float)
    
    def roi_intensity_stats(self):
        if not self._check_loaded():
            return
    
        if self.roi_mgr.last_roi_mask is None:
            QMessageBox.warning(self, "ROI Stats", "Please select an ROI first.")
            return
    
        z = self.roi_mgr.last_roi_z
        if z is None:
            QMessageBox.warning(self,
                "ROI Stats","ROI is not linked to z-plane. Generate image and reselect ROI.")
            return
    
        z = int(z)
        mask = self.roi_mgr.last_roi_mask
    
        try:
            self.statusbar.showMessage("📊 ROI Stats...")
    
            mz_list = self._get_available_mz_list()
            lines = []
    
            for (mz, tol) in mz_list:
                vals = self._current_roi_values_for_mz(mask, mz, tol, z)
                n = int(vals.size)
    
                mean = float(np.mean(vals)) if n else np.nan
                med = float(np.median(vals)) if n else np.nan
                std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                vmin = float(np.min(vals)) if n else np.nan
                vmax = float(np.max(vals)) if n else np.nan
                vsum = float(np.sum(vals)) if n else np.nan
    
                label = "all" if mz in (None, "ALL") else f"{mz:.4f} ± {tol}"
                lines.append(
                    f"m/z {label} | n={n} | mean={mean:.4g} | median={med:.4g} | "
                    f"std={std:.4g} | min={vmin:.4g} | max={vmax:.4g} | sum={vsum:.4g}")    
            QMessageBox.information(
                self,
                "ROI Stats",
                "ROI Intensity Stats\n\n" + "\n".join(lines))
    
            self.statusbar.showMessage("✅ ROI Stats computed")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ ROI Stats Error", str(e))
            self.statusbar.showMessage("❌ ROI Stats error")

    def _get_available_mz_list(self):
        mz_list = []   
        try:
            mz = float(self.targetA_textbox.text())
            tol = float(self.width_textbox.text())
            mz_list.append((mz, tol))
        except Exception:
            pass    
        if self.multiCheck.isChecked():
            try:
                mz2 = float(self.targetB_textbox.text())
                tol2 = float(self.width_textboxb.text())
                mz_list.append((mz2, tol2))
            except Exception:
                pass    
        if not mz_list:
            mz_list = [("ALL", None)]    
        return mz_list
    
    def save_current_roi(self):
        mask = self.roi_mgr.last_roi_mask
        if mask is None:
            QMessageBox.warning(self, "Save ROI", "Please select an ROI first.")
            return
    
        npx = int(mask.sum())
        if npx < 50:
            QMessageBox.warning(
                self,
                "Save ROI",
                f"ROI too small ({npx} pixels). Please select a larger region."
            )
            return
    
        z = self.roi_mgr.last_roi_z
        if z is None:
            QMessageBox.warning(self, "Save ROI", "ROI is not linked to z-plane.")
            return
    
        default_name = f"ROI_{self._roi_counter:02d}"
        name, ok = QInputDialog.getText(self, "Save ROI", "ROI name:", text=default_name)
        if not ok or not name.strip():
            name = default_name
    
        name = name.strip()
        if name in self.roi_mgr.saved_rois:
            QMessageBox.warning(
                self, "Save ROI",
                "ROI name already exists. Please choose a new name."
            )
            return
    
        # ---------- Shape guard ----------
        max_y = self.parser.imzmldict["max count of pixels y"]
        max_x = self.parser.imzmldict["max count of pixels x"]
        if mask.shape != (max_y, max_x):
            QMessageBox.critical(
                self, "ROI Shape Error",
                f"ROI mask shape {mask.shape} != parser dims {(max_y, max_x)}.\n"
                "Reload file and re-draw ROI."
            )
            return



    
        # ---------- SAVE ROI ----------
        self.roi_mgr.add_roi(name=name, mask=mask.copy(), z=z)
        self._roi_counter += 1
        # ---------- UI ----------
        self.roiList.addItem(name)
        self.comboROI_A.addItem(name)
        self.comboROI_B.addItem(name)
    
        # ================= SAVE ROI PATCH + LABEL =================
        import re
        
        m = re.match(r"ROI_(\d+)", name)
        if m:
            idx = int(m.group(1))
            self._roi_counter = max(self._roi_counter, idx + 1)

        # ================= SAVE ROI PATCH + LABEL =================
        w = self._active_image_widget()
        ax = w.canvas.ax
        
        if not hasattr(w, "saved_roi_patches"):
            w.saved_roi_patches = {}
        if not hasattr(w, "roi_labels"):
            w.roi_labels = {}
        
        # ---- ROI sınırlarından bounding box hesapla ----
        rr, cc = np.where(mask)
        if rr.size > 0:
            rmin, rmax = rr.min(), rr.max()
            cmin, cmax = cc.min(), cc.max()
        
            # 🔑 CROP OFFSET (AU dahil her dataset için şart)
            row0 = getattr(self, "_crop_row0", 0)
            col0 = getattr(self, "_crop_col0", 0)
        
            x0 = cmin - col0
            y0 = rmin - row0
            width  = cmax - cmin
            height = rmax - rmin
        
            # ---------- RECTANGLE ----------
            rect = Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="#16a34a",
                linewidth=2.2,
                zorder=9
            )
            ax.add_patch(rect)
            w.saved_roi_patches[name] = rect
        
            # ---------- LABEL ----------
            offset = 6
            label_x = (cmax - col0) + offset
            label_y = ((rmin + rmax) / 2) - row0
        
            ny, nx = w.roi_full_shape
            if label_x >= nx:
                label_x = (cmin - col0) - offset
                ha = "right"
            else:
                ha = "left"
        
            txt = ax.text(
                label_x,
                label_y,
                name,
                fontsize=11,
                fontweight="bold",
                ha=ha,
                va="center",
                zorder=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                    boxstyle="round,pad=0.25"
                )
            )
        
            w.roi_labels[name] = txt
            w.canvas.draw_idle()
        
        


    def toggle_roi_mask_overlay(self):
        if not self._check_loaded():
            return
        if self.roi_mgr.last_roi_mask is None:
            QMessageBox.warning(self, "Overlay Mask", "Please select an ROI first.")
            return
    
        # 🔥 KRİTİK FIX
        w = self._active_image_widget() 
    
        if w.roi_row_idx is None or w.roi_col_idx is None:
            QMessageBox.warning(self, "Overlay Mask", "Please generate an image first.")
            return
    
        ax = w.canvas.ax
    
        try:
            # remove if exists
            if w.roi_mask_overlay is not None:
                try:
                    w.roi_mask_overlay.remove()
                except Exception:
                    pass
                w.roi_mask_overlay = None
                w.canvas.draw_idle()
                self.statusbar.showMessage("🧩 Overlay removed")
                return
    
            mask = self.roi_mgr.last_roi_mask
            cropped_mask = mask[np.ix_(w.roi_row_idx, w.roi_col_idx)]    
            w.roi_mask_overlay = ax.imshow(
                cropped_mask,
                cmap="Reds",
                alpha=0.35,
                interpolation="nearest",
                aspect="auto")
            w.canvas.draw_idle()
            self.statusbar.showMessage("🧩 Overlay added")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Overlay Error", str(e))
    def roi_similarity(self):
        if not self._check_loaded():
            return
    
        # ROI seçimi
        selected = [i.text() for i in self.roiList.selectedItems()]
    
        if len(selected) < 2:
            nameA = self.comboROI_A.currentText().strip()
            nameB = self.comboROI_B.currentText().strip()
            if nameA and nameB and nameA != nameB and \
               (nameA in self.roi_mgr.saved_rois) and (nameB in self.roi_mgr.saved_rois):
                selected = [nameA, nameB]
    
        if len(selected) < 2:
            QMessageBox.warning(self, "ROI Similarity", "Please select two ROIs.")
            return
    
        nameA, nameB = selected[:2]
    
        infoA = self.roi_mgr.get_roi(nameA)
        infoB = self.roi_mgr.get_roi(nameB)
    
        if infoA is None or infoB is None:
            QMessageBox.warning(self, "ROI Similarity", "ROI data not found.")
            return
    
        zA, zB = int(infoA["z"]), int(infoB["z"])
        if zA != zB:
            QMessageBox.warning(
                self,
                "ROI Similarity",
                f"Z-plane mismatch: {nameA} (z={zA}) vs {nameB} (z={zB})")
            return
    
        maskA = infoA["mask"].astype(bool)
        maskB = infoB["mask"].astype(bool)
    
        # Dice & Jaccard
        intersection = np.logical_and(maskA, maskB).sum()
        areaA = maskA.sum()
        areaB = maskB.sum()
        union = np.logical_or(maskA, maskB).sum()
    
        if areaA == 0 or areaB == 0:
            QMessageBox.warning(self, "ROI Similarity", "One of the ROIs is empty.")
            return
    
        dice = 2 * intersection / (areaA + areaB)
        jaccard = intersection / union if union > 0 else 0.0
    
        QMessageBox.information(
            self,
            "ROI Similarity",
            f"ROI A: {nameA}\n"
            f"ROI B: {nameB}\n\n"
            f"Dice coefficient: {dice:.3f}\n"
            f"Jaccard index: {jaccard:.3f}"
        )
    
        self.statusbar.showMessage(
            f"ROI Similarity → Dice={dice:.3f}, Jaccard={jaccard:.3f}"
        )
    def _get_plot_ax(self, plot_kind):
        """
        plot_kind: 'box', 'violin', 'volcano'
        """
        if plot_kind == "box":
            return self.boxPlot.canvas.ax, self.boxPlot
        elif plot_kind == "violin":
            return self.multiPlot.canvas.ax, self.multiPlot
        elif plot_kind == "volcano":
            return self.volcanoPlot.canvas.ax, self.volcanoPlot
        else:
            raise ValueError(f"Unknown plot kind: {plot_kind}")
            

    def export_roi_csv(self):
        if not self._check_loaded():
            return
    
        if len(self.roi_mgr.saved_rois) == 0:
            QMessageBox.warning(
                self,
                "Export ROI CSV",
                "No saved ROIs. Use 'Save ROI' first."
            )
            return
    
        selected = [i.text() for i in self.roiList.selectedItems()]
        roi_names = selected if selected else list(self.roi_mgr.keys())
    
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI CSV",
            "roi_statistics.csv",
            "CSV (*.csv)"
        )
        if not fname:
            return
    
        try:
            self.statusbar.showMessage("🧾 Exporting ROI statistics...")
    
            mz1, tol1, mz2, tol2, mz3, tol3, _z = self._read_mz_block()
            mz_list = [(mz1, tol1)]
            if self.multiCheck.isChecked():
                mz_list.extend([(mz2, tol2), (mz3, tol3)])
    
            with open(fname, "w", newline="", encoding="utf-8") as f:
                # ---------- METADATA HEADER ----------
                f.write("# Software: PyMSIViz v2.x\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write("# Intensity: raw (not log-transformed)\n")
                f.write("# ROI-based pixel statistics\n")
                f.write("# Outliers: not removed (reported only)\n")
                f.write("# ------------------------------------\n")
    
                writer = csv.writer(f)
                writer.writerow([
                    "roi_name", "z",
                    "mz", "tol",
                    "n_pixels",
                    "mean", "median", "iqr", "std",
                    "min", "max", "sum",
                    "outlier_ratio",
                    "test_used",
                    "notes"
                ])
    
                for name in roi_names:
                    info = self.roi_mgr.get_roi(name)
                    if info is None:
                        continue
    
                    mask = info["mask"]
                    z = int(info["z"])
    
                    for (mz, tol) in mz_list:
                        vals = self._current_roi_values_for_mz(mask, mz, tol, z)
                        vals = vals[np.isfinite(vals)]
                        n = int(vals.size)
    
                        if n == 0:
                            continue
    
                        mean = float(np.mean(vals))
                        med = float(np.median(vals))
                        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                        vmin = float(np.min(vals))
                        vmax = float(np.max(vals))
                        vsum = float(np.sum(vals))
    
                        q1, q3 = np.percentile(vals, [25, 75])
                        iqr = float(q3 - q1)
                        outlier_ratio = float(np.mean(vals > q3 + 3 * iqr)) if iqr > 0 else 0.0
    
                        writer.writerow([
                            name, z,
                            f"{mz:.5f}", tol,
                            n,
                            mean, med, iqr, std,
                            vmin, vmax, vsum,
                            outlier_ratio,
                            "Mann–Whitney U (pairwise)",
                            "no outlier removal"
                        ])
    
            QMessageBox.information(self, "✅ Exported", f"Saved:\n{fname}")
            self.statusbar.showMessage("✅ ROI CSV exported")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Export ROI CSV Error", str(e))
            self.statusbar.showMessage("❌ Export ROI CSV error")
    
    # --------------------------- Export / Clear ---------------------------
    def export_current_tab(self):
        fig = None
        default_name = "figure.png"
    
        outer = self.leftTabs.currentIndex()
    
        if outer == 0:  # Spectrum
            fig = self.spectrumPlot.canvas.fig
            default_name = "spectrum.png"
    
        elif outer == 1:  # Image
            idx = self.imageTabs.currentIndex()
            figs = [
                self.image1Plot.canvas.fig,
                self.image2Plot.canvas.fig,
                self.image3Plot.canvas.fig
            ]
            fig = figs[idx] if 0 <= idx < len(figs) else None
            default_name = f"image_{idx+1}.png"
    
        elif outer == 2:  # Plot
            idx = self.plotTabs.currentIndex()
            figs = [
                self.boxPlot.canvas.fig,      # Plot 1
                self.multiPlot.canvas.fig,    # Plot 2
                self.posthocPlot.canvas.fig,  # Plot 3 (Multi-ROI)
                self.volcanoPlot.canvas.fig   # Plot 4 (Volcano)
            ]
            fig = figs[idx] if 0 <= idx < len(figs) else None
            default_name = f"plot_{idx+1}.png"
    
        if fig is None:
            QMessageBox.warning(self, "Export", "No figure available to export.")
            return
    
        fname, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            default_name,
            "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)")
    
        if not fname:
            return

        ext = os.path.splitext(fname)[1].lower()
        if not ext:
            if "PNG" in selected_filter:
                fname += ".png"
            elif "JPEG" in selected_filter:
                fname += ".jpg"
            elif "PDF" in selected_filter:
                fname += ".pdf"
            elif "SVG" in selected_filter:
                fname += ".svg"
    
        try:
            fig.savefig(
                fname,
                dpi=300,
                bbox_inches="tight",
                facecolor="white"
            )
            QMessageBox.information(self, "✅ Saved", f"Exported:\n{fname}")
            self.statusbar.showMessage(f"✅ Exported: {os.path.basename(fname)}")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
   
    def _reset_for_new_file(self):
        # 1) ROI ve ROI UI
        try:
            self.clear_roi()
        except Exception:
            pass
    
        try:
            self.roi_mgr.clear()
            self.roiList.clear()
            self.comboROI_A.clear()
            self.comboROI_B.clear()
            self._roi_counter = 1
        except Exception:
            pass
    
        try:
            self._ion_cache.clear()
            self._ion_cache_order.clear()
        except Exception:
            pass

        widgets = [
            self.spectrumPlot,
            self.image1Plot,
            self.image2Plot,
            self.image3Plot,
            self.boxPlot,
            self.multiPlot,
            self.volcanoPlot,]
    
        for w in widgets:
            try:
                w.canvas.ax.clear()
                if w is self.spectrumPlot:
                    w.canvas.ax.set_axis_on()   
                else:
                    w.canvas.ax.set_axis_off()
            except Exception:
                pass
    
            try:
                if getattr(w, "cbar", None) is not None:
                    w.cbar.remove()
                w.cbar = None
            except Exception:
                pass
    
            try:
                if getattr(w, "cax", None) is not None:
                    w.cax.remove()
                w.cax = None
            except Exception:
                pass
    
            try:
                if getattr(w, "roi_overlay", None) is not None:
                    w.roi_overlay.remove()
                w.roi_overlay = None
            except Exception:
                pass
    
            try:
                if getattr(w, "roi_mask_overlay", None) is not None:
                    w.roi_mask_overlay.remove()
                w.roi_mask_overlay = None
            except Exception:
                pass

            try:
                w.last_full_image = None
                w.last_mz = None
                w.last_tol = None
            except Exception:
                pass
    
            try:
                w.canvas.draw_idle()
            except Exception:
                pass

        try:
            self.targetA_textbox.clear()
            self.width_textbox.clear()
            self.z_textbox.clear()
            self._global_spectrum_cache.clear()
    
            self.targetB_textbox.clear()
            self.width_textboxb.clear()
            self.targetC_textbox.clear()
            self.width_textboxc.clear()
        except Exception:
            pass
    
        try:
            self.multiCheck.setChecked(False)
        except Exception:
            pass

        try:
            self._clear_feature_cache()
        except Exception:
            pass
    
        # 🔥 ROI VALUE CACHE (EN KRİTİK SATIR)
        try:
            self._roi_value_cache.clear()
        except Exception:
            pass
    
    def clear_all(self):
        # ---- ROI ----
        try:
            self.clear_roi()
        except Exception:
            pass

        widgets = [
            self.spectrumPlot,
            self.image1Plot,
            self.image2Plot,
            self.image3Plot,
            self.boxPlot,
            self.multiPlot,
            self.volcanoPlot,]
    
        for w in widgets:
            try:
                ax = w.canvas.ax
                ax.clear()
                ax.set_axis_off()
                w.canvas.draw_idle()
            except Exception:
                pass

            try:
                if getattr(w, "cbar", None):
                    w.cbar.remove()
                    w.cbar = None
            except Exception:
                pass
    
        try:
            self.roi_mgr.clear()
            self.roiList.clear()
            self.comboROI_A.clear()
            self.comboROI_B.clear()
            self._roi_counter = 1
        except Exception:
            pass
    
        for t in [
            self.targetA_textbox, self.width_textbox, self.z_textbox,
            self.targetB_textbox, self.width_textboxb,
            self.targetC_textbox, self.width_textboxc,
        ]:
            try:
                t.clear()
            except Exception:
                pass
        try:
            self._ion_cache.clear()
            self._roi_value_cache.clear()
            self._clear_feature_cache()
        except Exception:
            pass
    
        self.statusbar.showMessage("🧹 Cleared")

    def show_about(self):
        QMessageBox.about(
            self, 
            "About PyMSIViz",
            "<h2>PyMSIViz_v2</h2>"
            "<p>Responsive Modern MSI Visualization Tool + ROI + AI Assist</p>"
            "<ul>"
            "<li>Tabs: Spectrum / Image / Plot</li>"
            "<li>AI Assist: Top Peak → Image 1</li>"
            "<li>Multi optional: m/z2–m/z3</li>"
            "<li>ROI: Rectangle + Lasso</li>"
            "<li>ROI Stats + Save ROI + Overlay + Corr + Compare + Export CSV</li>"
            "</ul>"
            "<li>NEW: Violin plot (single & multi)</li>"
            "<li>NEW: ROI Compare (Violin)</li>"
            "<li>NEW: Volcano plot (ROI differential MSI)</li>"
            "</ul>"
        )
    def export_benchmark_csv(self):
        if not getattr(self, "benchmark_log", None):
            QMessageBox.information(self, "Benchmark", "No benchmark records yet.")
            return
    
        default_name = f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Benchmark CSV",
            os.path.join(os.getcwd(), default_name),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
    
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "file",
                    "pixels",
                    "size_mb"])

                for r in self.benchmark_log:
                    w.writerow([
                        r.get("time", ""),
                        r.get("file", ""),
                        r.get("pixels", ""),
                        r.get("size_mb", "")])
            self.statusbar.showMessage(f"✅ Benchmark CSV exported: {os.path.basename(path)}")
            print("[BENCH] Exported:", path)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
       
    def clear_benchmark_log(self):
        if not getattr(self, "benchmark_log", None):
            QMessageBox.information(self, "Benchmark", "Benchmark log is already empty.")
            return
        self.benchmark_log.clear()
        self.statusbar.showMessage("🧹 Benchmark log cleared.")
        print("[BENCH] Log cleared.")
    
    def closeEvent(self, event):
        # Çalışan worker varsa iptal et
        if self._running_worker is not None:
            try:
                self._running_worker.cancel()
            except Exception:
                pass
            self._running_worker = None
    
        event.accept()
        QApplication.quit() 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI", 10))
    else:
        # macOS/Linux: sistem fontu kalsın (en stabil seçenek)
        app.setFont(QFont("", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
