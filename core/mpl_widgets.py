from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        super().__init__(self.fig)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- CANVAS ---
        self.canvas = MplCanvas()

        # --- TOOLBAR (TEK VE SADECE TEK KEZ) ---
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(18, 18))
        self.toolbar.setStyleSheet("background: white; border: none;")

        # --- OPTIONAL STATE ---
        self.cbar = None
        self.roi_overlay = None
        self.roi_mask_overlay = None

        self.last_full_image = None
        self.last_mz = None
        self.last_tol = None

        # --- LAYOUT ---
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setStyleSheet("background: white;")
        self.setAutoFillBackground(True)



