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
    from scipy.signal import find_peaks, savgol_filter
    from scipy.stats import mannwhitneyu
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
        self.setWindowTitle("PyMSIViz v2.6 - Responsive + ROI + AI Assist")
        self.resize(1200, 780)
        self.setMinimumSize(980, 620)
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
    
        df_mz, df_int = self._compute_global_spectrum_binned(max_pixels=int(max_pixels),n_bins=int(n_bins), stat="mean")    
        feats = self._peak_pick_features(df_mz, df_int, top_k=int(top_k),
                                         min_mz_dist=float(min_mz_dist),
                                         snr=float(snr))
        if feats.size < 30:
            top_idx = np.argsort(df_int)[-min(300, df_mz.size):]
            feats = np.unique(np.sort(df_mz[top_idx].astype(float)))
    
        self._feature_cache = feats
        self._feature_cache_key = key
        return feats 
 
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
    
        self.plotTabs.addTab(self.boxPlot, "Plot 1")
        self.plotTabs.addTab(self.multiPlot, "Plot 2")
        self.plotTabs.addTab(self.volcanoPlot, "Plot 3")
        self.plotTabs.setTabEnabled(2, False)
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
        
        self.multiCheck.toggled.connect(self._update_image_processing_state)
        self.buttonBox.clicked.connect(self.run_selected_plot)
        self.cb.currentIndexChanged.connect(self._update_image_processing_state)
        self.cb2.currentIndexChanged.connect(self._update_image_processing_state)
        #self.btnCancel.clicked.connect(self.cancel_running_task)
        self.clearButton.clicked.connect(self.clear_all)
        self.closeButton.clicked.connect(self.ask_quit)

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






    def _on_top_peaks_toggled(self, checked):
        self.enable_top_peaks = bool(checked)
    
        if checked:
            self.statusbar.showMessage("🧠 Top peak detection ENABLED")
        else:
            self.statusbar.showMessage("⛔ Top peak detection DISABLED")

    def _add_mz_subtitle(self, ax, text):
        ax.text(
            0.5, 0.93,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            color="#4b5563")


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
            self.statusbar.showMessage("📦 ROI Compare (Boxplot)...")
    
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
    
            ax, widget = self._get_plot_ax("box")
            ax.clear()
    
            box = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False
            )
    
            for b in box["boxes"]:
                b.set_facecolor("#93c5fd")
                b.set_edgecolor("#2563eb")
            for m in box["medians"]:
                m.set_color("#111827")
                m.set_linewidth(2)
    
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["ROI 1", "ROI 2"], fontsize=10)
            ax.tick_params(axis="x", rotation=0)
    
            self._apply_axes_style(
                ax,
                title="ROI 1 vs ROI 2 Comparison (Boxplot)",
                ylabel="Intensity"
            )
    
            self._add_mz_subtitle(ax, f"m/z {mz1:.4f} ± {tol1}")
    
            widget.canvas.draw_idle()
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(0)
            self.statusbar.showMessage("✅ ROI Compare (Boxplot) ready")
    
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
            self.statusbar.showMessage("📦 ROI Compare (Violin)...")
    
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
    
            ax = self.multiPlot.canvas.ax
            ax.clear()
    
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
    
            if "cmedians" in parts:
                parts["cmedians"].set_color("#111827")
                parts["cmedians"].set_linewidth(2)
    
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["ROI 1", "ROI 2"], fontsize=10)
            ax.tick_params(axis="x", rotation=0)
    
            self._apply_axes_style(
                ax,
                title="ROI 1 vs ROI 2 Comparison (Violin)",
                ylabel="Intensity"
            )
    
            self._add_mz_subtitle(ax, f"m/z {mz1:.4f} ± {tol1}")
    
            self.multiPlot.canvas.draw_idle()
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(1)
            self.statusbar.showMessage("✅ ROI Compare (Violin) ready")
    
        except Exception as e:
            QMessageBox.critical(self, "❌ ROI Compare (Violin) Error", str(e))


            
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
    
        
        def _on_result(payload):
            ax = self.volcanoPlot.canvas.ax
            ax.clear()
        
            if payload is None or payload["log2fc"].size == 0:
                ax.text(0.5, 0.5, "No significant features found",
                        ha="center", va="center", fontsize=11)
                ax.set_axis_off()
            else:
                ax.scatter(
                    payload["log2fc"],
                    payload["y"],
                    c=payload["rejected"],
                    cmap="coolwarm",
                    alpha=0.8
                )
                ax.axhline(-np.log10(0.05), linestyle="--", color="gray")
                ax.axvline(0, linestyle="--", color="gray")
        
                self._apply_axes_style(
                    ax,
                    title="ROI Volcano (ROI 1 vs ROI 2)",
                    xlabel="log2 Fold Change",
                    ylabel="-log10(p-value)"
                )
        
                self._add_mz_subtitle(
                    ax,
                    "Feature-wise ROI comparison (exploratory)"
                )
        
            self.volcanoPlot.canvas.draw_idle()
            self.plotTabs.setTabEnabled(2, True)
            self.leftTabs.setCurrentIndex(2)
            self.plotTabs.setCurrentIndex(2)
            self.statusbar.showMessage("✅ Volcano finished")

        # ================= ERROR =================
        def _on_error(msg):
            print("[VOLCANO ERROR]", msg)
            QMessageBox.critical(self, "Volcano Error", str(msg))
            self.statusbar.showMessage("❌ Volcano failed")
    
        worker.result.connect(_on_result)
        worker.error.connect(_on_error)
    
        self._running_worker = worker
        self.worker_mgr.start(worker)

# --------------------------- Style helpers ---------------------------

    def _apply_axes_style(self, ax, title=None, xlabel=None, ylabel=None):
        ax.set_facecolor("#ffffff")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=10)
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12)

    def _on_load_done(self, payload):
        parser, info = payload
        self.parser = parser
    
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
        total_mb = getattr(self, "last_total_size_mb", 0)
        total_bytes = int(total_mb * 1024**2)
        size_str = self._format_bytes(total_bytes)
    
        pixels = len(parser.coordinates)
        z_planes = len(set(z for (_, _, z) in parser.coordinates)) if parser.coordinates else 0
    
        mode = "Large Dataset" if self.large_dataset_mode else "Standard"
    
        print("\n📦 Dataset Info")
        print("-" * 30)
        print(f"File: {os.path.basename(self.imzml_filename)}")
        print(f"Pixels: {pixels:,}")
        print(f"Z-planes: {z_planes}")
        print(f"m/z range: {info['mz_min']:.2f} – {info['mz_max']:.2f}")
        print(f"Size: {size_str}")
        print(f"Mode: {mode}\n")

        from datetime import datetime
        
        self.benchmark_log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": os.path.basename(self.imzml_filename),
            "pixels": pixels,
            "size_mb": round(self.last_total_size_mb, 2),
            "load_s": round(self.last_load_time_s or 0.0, 3),
            "ram_mb": round(self.last_peak_ram_mb or 0.0, 1),
        })
    
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
  

    def _compute_global_spectrum_binned(self, max_pixels=1200, n_bins=6000, stat="mean"):
        return compute_global_spectrum_binned(self.parser,max_pixels=max_pixels,n_bins=n_bins,stat=stat)    

    def process_spectrum(self):
        if not self._check_loaded():
            return
    
        try:
            
            max_pixels = 600 if self.large_dataset_mode else 2000
            
            df_mz, df_int = self._compute_global_spectrum_binned(
                max_pixels=max_pixels,
                n_bins=6000,
                stat="mean"
            )            
    
            ax = self.spectrumPlot.canvas.ax
            ax.clear()
            ax.set_axis_on()
            ax.plot(df_mz, df_int, linewidth=1.2)

            if hasattr(self, "_scatter_top_peaks") and self._scatter_top_peaks:
                try:
                    self._scatter_top_peaks.remove()
                except Exception:
                    pass
                self._scatter_top_peaks = None

            if hasattr(self, "chkTopPeaks") and self.chkTopPeaks.isChecked():  
                try:
                    from scipy.signal import find_peaks
                except Exception:
                    self.statusbar.showMessage("⚠️ SciPy not available (peak detection skipped)")
                    find_peaks = None
    
                if find_peaks is not None:
                    N = 5  # ileride SpinBox bağlanabilir
                    peaks, props = find_peaks(df_int,distance=20,prominence=np.max(df_int) * 0.01)    
                    if len(peaks) > 0:
                        top_idx = np.argsort(df_int[peaks])[-N:]
                        self._top_peaks_idx = peaks[top_idx]
                        self._top_peaks_mz = df_mz[self._top_peaks_idx]
                        self._top_peaks_int = df_int[self._top_peaks_idx]
    
                        # --- SCATTER ---
                        self._scatter_top_peaks = ax.scatter(
                            self._top_peaks_mz,
                            self._top_peaks_int,
                            color="red",s=40,zorder=5,
                            label=f"Top {len(self._top_peaks_mz)} peaks",
                            picker=True,
                            pickradius=5)
    
                        # --- VERTICAL LINES ---
                        for mz in self._top_peaks_mz:
                            ax.axvline(
                                mz,
                                color="red",
                                alpha=0.25,
                                linestyle="--")
                        ax.legend()
                    else:
                        self._top_peaks_idx = []
                        self._top_peaks_mz = []
                        self._top_peaks_int = []
    
            else:
                self._top_peaks_idx = []
                self._top_peaks_mz = []
                self._top_peaks_int = []    
            # --- STYLING ---
            self._apply_axes_style(
                ax,
                title="Global Mean Spectrum (binned grid, sampled)",
                xlabel="m/z",
                ylabel="Intensity")    
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

            max_pixels = 600 if self.large_dataset_mode else 2000
            
            df_mz, df_int = self._compute_global_spectrum_binned(
                max_pixels=max_pixels,
                n_bins=6000,
                stat="mean"
            )
    
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

    def start_roi_rectangle(self):
        """
        Start rectangular ROI selection on active image.
        """
        if not self._check_loaded():
            return
        try:
            self._start_roi_mode(mode="rect")
        except AttributeError:
            QMessageBox.warning(
                self,
                "ROI",
                "ROI rectangle mode is not available."
            )
    
    
    def start_roi_lasso(self):
        """
        Start freehand (lasso) ROI selection on active image.
        """
        if not self._check_loaded():
            return
        try:
            self._start_roi_mode(mode="lasso")
        except AttributeError:
            QMessageBox.warning(
                self,
                "ROI",
                "ROI lasso mode is not available."
            )



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
        mpl_widget.cbar.ax.tick_params(labelsize=9)  # sağdaki 0-100 gibi sayılar
        
        mpl_widget.cbar.set_label( "Intensity", fontsize=11,fontweight="bold",labelpad=10)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    
        # store mapping for ROI
        mpl_widget.roi_row_idx = row_idx
        mpl_widget.roi_col_idx = col_idx
        mpl_widget.roi_full_shape = full_shape
        mpl_widget.roi_z = z_used
    
        # store last image + params
        mpl_widget.last_full_image = full_img
        mpl_widget.last_mz = mz
        mpl_widget.last_tol = tol
    
        # drop overlays (new image)
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

    def clear_roi(self):
        self._disconnect_roi_selector()
        self.roi_mgr.last_roi_mask = None
        self.roi_mgr.last_roi_z = None
        self.lblROIst.setText("ROI: none")

        for w in [self.image1Plot, self.image2Plot, self.image3Plot]:
            if w.roi_overlay is not None:
                try:
                    w.roi_overlay.remove()
                except Exception:
                    pass
                w.roi_overlay = None

            if w.roi_mask_overlay is not None:
                try:
                    w.roi_mask_overlay.remove()
                except Exception:
                    pass
                w.roi_mask_overlay = None

            w.canvas.draw()
            self.roi_dirty = True

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

            xmin_i = max(0, xmin_i); ymin_i = max(0, ymin_i)
            xmax_i = min(len(w.roi_col_idx) - 1, xmax_i)
            ymax_i = min(len(w.roi_row_idx) - 1, ymax_i)

            full_cols = w.roi_col_idx[xmin_i:xmax_i+1]
            full_rows = w.roi_row_idx[ymin_i:ymax_i+1]

            mask = np.zeros(w.roi_full_shape, dtype=bool)
            mask[np.ix_(full_rows, full_cols)] = True

            self.roi_mgr.last_roi_mask = mask
            self.roi_mgr.last_roi_z = w.roi_z            
            self.roi_dirty = True
            if w.roi_overlay is not None:
                try:
                    w.roi_overlay.remove()
                except Exception:
                    pass
                w.roi_overlay = None

            rect = Rectangle((xmin_i, ymin_i),
                             (xmax_i - xmin_i),
                             (ymax_i - ymin_i),
                             fill=False, linewidth=2.0, edgecolor="red")
            w.roi_overlay = ax.add_patch(rect)
            w.canvas.draw()

            npx = int(mask.sum())
            self.lblROIst.setText(f"ROI: Rectangle | pixels={npx} | z={w.roi_z}")

        self.roi_selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
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
            if w.roi_overlay is not None:
                try:
                    w.roi_overlay.remove()
                except Exception:
                    pass
                w.roi_overlay = None

            patch = PathPatch(path, fill=False, linewidth=2.0, edgecolor="red")
            w.roi_overlay = ax.add_patch(patch)
            w.canvas.draw()

            npx = int(mask.sum())
            self.lblROIst.setText(f"ROI: Lasso | pixels={npx} | z={w.roi_z}")

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
        
        if self.roi_mgr.last_roi_mask is None:
            QMessageBox.warning(self, "Save ROI", "Please select an ROI first.")
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
            QMessageBox.warning(self, "Save ROI", "ROI name already exists. Please choose a new name.")
            return            
        # shape guard (z-plane mismatch / parser mismatch early detection)
        max_y = self.parser.imzmldict["max count of pixels y"]
        max_x = self.parser.imzmldict["max count of pixels x"]
        if self.roi_mgr.last_roi_mask.shape != (max_y, max_x):
            QMessageBox.critical(
                self, "ROI Shape Error",
                f"ROI mask shape {self.roi_mgr.last_roi_mask.shape} != parser dims {(max_y, max_x)}.\n"
                "This usually happens if ROI was created from a different dataset/session.\n"
                "Reload file and re-draw ROI." )
            return
        self.roi_mgr.add_roi(name=name, mask=self.roi_mgr.last_roi_mask.copy(),z=self.roi_mgr.last_roi_z)    
        self.roiList.addItem(name)       
        item = self.roiList.item(self.roiList.count() - 1)
        item.setSelected(True)
        self.roiList.setCurrentItem(item)
        self.roiList.scrollToItem(item)
        
        self.comboROI_A.addItem(name)
        self.comboROI_B.addItem(name)
        
        if self.comboROI_A.count() == 1:
            self.comboROI_A.setCurrentIndex(0)
        if self.comboROI_B.count() == 2:
            self.comboROI_B.setCurrentIndex(1)
        
        if self.roiList.count() >= 2:
            for i in range(self.roiList.count()):
                self.roiList.item(i).setSelected(False)
            self.roiList.item(self.roiList.count()-1).setSelected(True)
            self.roiList.item(self.roiList.count()-2).setSelected(True)
        
        self._roi_counter += 1
        self.statusbar.showMessage(f"Saved ROI: {name}")
        self.roi_dirty = True 

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
            QMessageBox.warning(self, "Export ROI CSV", "No saved ROIs. Use 'Save ROI' first.")
            return

        selected = [i.text() for i in self.roiList.selectedItems()]
        roi_names = selected if selected else list(self.roi_mgr.keys())

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export ROI CSV", "roi_stats.csv", "CSV (*.csv)"
        )
        if not fname:
            return

        try:
            self.statusbar.showMessage("🧾 Exporting ROI CSV...")

            mz1, tol1, mz2, tol2, mz3, tol3, _z = self._read_mz_block()
            mz_list = [(mz1, tol1)]
            if self.multiCheck.isChecked():
                mz_list.extend([(mz2, tol2), (mz3, tol3)])

            with open(fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["roi_name", "z", "mz", "tol", "n_pixels", "mean", "median", "std", "min", "max", "sum"])

                for name in roi_names:
                    info = self.roi_mgr.get_roi(name)
                    if info is None:
                        continue
                    mask = info["mask"]
                    z = int(info["z"])

                    for (mz, tol) in mz_list:
                        vals = self._current_roi_values_for_mz(mask, mz, tol, z)
                        n = int(vals.size)
                        mean = float(np.mean(vals)) if n else np.nan
                        med = float(np.median(vals)) if n else np.nan
                        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                        vmin = float(np.min(vals)) if n else np.nan
                        vmax = float(np.max(vals)) if n else np.nan
                        vsum = float(np.sum(vals)) if n else np.nan
                        writer.writerow([name, z, mz, tol, n, mean, med, std, vmin, vmax, vsum])

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
    
        else:  # Plot
            idx = self.plotTabs.currentIndex()
            figs = [
                self.boxPlot.canvas.fig,
                self.multiPlot.canvas.fig,
                self.volcanoPlot.canvas.fig
            ]
            fig = figs[idx] if 0 <= idx < len(figs) else None
            default_name = "plot.png"
    
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
    
