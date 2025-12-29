from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QGridLayout, QLineEdit, QLabel, QCheckBox)
from PyQt5.QtWidgets import QAbstractItemView
def build_group_datafile(self):
        self.groupData = QGroupBox("Data File")
        lay = QGridLayout(self.groupData)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(10)

        self.buttonLoadData = QPushButton("Load")
        self.buttonLoadData.setObjectName("DataButton")
        self.buttonLoadData.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.buttonLoadData.setMinimumWidth(78)
    
        self.viewFileName = QLineEdit()
        self.viewFileName.setReadOnly(True)
        self.viewFileName.setPlaceholderText("No file loaded...")    
        lay.addWidget(self.buttonLoadData, 0, 0)
        lay.addWidget(self.viewFileName, 0, 1, 1, 3)
    
        self.buttonSpectrum = QPushButton("Spectrum")
        self.buttonSpectrum.setObjectName("purpleButton")
        self.buttonSpectrum.setMinimumHeight(44)
        lay.addWidget(self.buttonSpectrum, 1, 0, 1, 4)
    

        self.chkTopPeaks = QCheckBox("Detect Top Peaks (AI-assisted)")
        self.chkTopPeaks.setChecked(False)
        lay.addWidget(self.chkTopPeaks, 2, 0, 1, 4)
        
        self.aiAssistBtn = QPushButton("AI Assist: Top Peak → Image 1")
        self.aiAssistBtn.setObjectName("greenButton")
        self.aiAssistBtn.setMinimumHeight(42)
        self.aiAssistBtn.setEnabled(False)
        lay.addWidget(self.aiAssistBtn, 3, 0, 1, 4)
    
def build_group_mz(self):
        self.groupMz = QGroupBox("m/z Parameters")
        lay = QGridLayout(self.groupMz)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(8)
    
        validator = QDoubleValidator()
         
        self.targetA_textbox = QLineEdit()
        self.width_textbox = QLineEdit()
        self.z_textbox = QLineEdit()
    
        self.targetB_textbox = QLineEdit()
        self.width_textboxb = QLineEdit()
        self.targetC_textbox = QLineEdit()
        self.width_textboxc = QLineEdit()
    
        for w in [self.targetA_textbox, self.width_textbox, self.z_textbox,
                  self.targetB_textbox, self.width_textboxb, self.targetC_textbox, self.width_textboxc]:
            w.setValidator(validator)
    
        self.lbl_mz1 = QLabel("m/z 1:")
        self.lbl_tol1 = QLabel("tol 1:")
        self.lbl_z = QLabel("z:")
    
        lay.addWidget(self.lbl_mz1, 0, 0); lay.addWidget(self.targetA_textbox, 0, 1)
        lay.addWidget(self.lbl_tol1, 1, 0); lay.addWidget(self.width_textbox, 1, 1)
        lay.addWidget(self.lbl_z, 2, 0); lay.addWidget(self.z_textbox, 2, 1)
   
        self.multiCheck = QCheckBox("Enable Multi (m/z2–m/z3)")
        self.multiCheck.setChecked(False)
        lay.addWidget(self.multiCheck, 3, 0, 1, 2)
    
        self.lbl_mz2 = QLabel("m/z 2:")
        self.lbl_tol2 = QLabel("tol 2:")
        self.lbl_mz3 = QLabel("m/z 3:")
        self.lbl_tol3 = QLabel("tol 3:")
    
        lay.addWidget(self.lbl_mz2, 4, 0); lay.addWidget(self.targetB_textbox, 4, 1)
        lay.addWidget(self.lbl_tol2, 5, 0); lay.addWidget(self.width_textboxb, 5, 1)
        lay.addWidget(self.lbl_mz3, 6, 0); lay.addWidget(self.targetC_textbox, 6, 1)
        lay.addWidget(self.lbl_tol3, 7, 0); lay.addWidget(self.width_textboxc, 7, 1)
    
        self._multi_row_widgets = [
            self.lbl_mz2, self.targetB_textbox,
            self.lbl_tol2, self.width_textboxb,
            self.lbl_mz3, self.targetC_textbox,
            self.lbl_tol3, self.width_textboxc
        ]    
        for w in self._multi_row_widgets:
            w.setVisible(False)    
          
        self.multiCheck.toggled.connect(self._set_multi_enabled)
        self._set_multi_enabled(self.multiCheck.isChecked())
        
def build_group_actions(self):
        self.groupActions = QGroupBox("Actions")
        lay = QVBoxLayout(self.groupActions)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setSpacing(12)

        rowCmap = QHBoxLayout()
        rowCmap.setSpacing(8)   
        lblCmap = QLabel("cmap:")
        lblCmap.setMinimumWidth(40)
        rowCmap.addWidget(lblCmap)    
        self.cb = QComboBox()
        self.cb.addItems(["jet","Spectral","viridis", "cividis",
                          "plasma","magma","inferno", "hot"])
        self._fix_combo_popup(self.cb)
        rowCmap.addWidget(self.cb, 1)   
        lay.addLayout(rowCmap)      

        rowInterp = QHBoxLayout()
        rowInterp.setSpacing(8)
        lblInterp = QLabel("interp:")
        lblInterp.setMinimumWidth(40)
        rowInterp.addWidget(lblInterp)    
        self.cb2 = QComboBox()
        self.cb2.addItems(["nearest", "bilinear", "bicubic", "hanning", "lanczos"])
        self._fix_combo_popup(self.cb2)
        rowInterp.addWidget(self.cb2, 1)
        lay.addLayout(rowInterp)

        self.buttonImage = QPushButton("Image Processing")
        self.buttonImage.setObjectName("greenButton")
        self.buttonImage.setMinimumHeight(42)
        self.buttonImage.setEnabled(False)
        lay.addWidget(self.buttonImage)

        rowPlotType = QHBoxLayout()
        rowPlotType.setSpacing(8)       
        lblPlotType = QLabel("Plot Type:")
        lblPlotType.setMinimumWidth(70)
        rowPlotType.addWidget(lblPlotType)
        
        self.plotTypeCombo = QComboBox()
        self.plotTypeCombo.addItems(["Box Plot", "Violin Plot"])
        self._fix_combo_popup(self.plotTypeCombo)
        rowPlotType.addWidget(self.plotTypeCombo, 1)      
        lay.addLayout(rowPlotType)
        
        self.buttonBox = QPushButton("Plot")
        self.buttonBox.setObjectName("greenButton")
        self.buttonBox.setMinimumHeight(42)
        self.buttonBox.setEnabled(False)        
        lay.addWidget(self.buttonBox)  
            
        row = QHBoxLayout()
        row.setSpacing(10)
        self.clearButton = QPushButton("Clear")
        self.clearButton.setObjectName("grayButton")
        self.clearButton.setMinimumHeight(42)
        
        self.closeButton = QPushButton("Close")
        self.closeButton.setObjectName("redButton")
        self.closeButton.setMinimumHeight(42)

        row.addWidget(self.clearButton)
        row.addWidget(self.closeButton)

        lay.addLayout(row)
        
def build_group_roi(self):
        self.groupROI = QGroupBox("ROI")
        lay = QVBoxLayout(self.groupROI)
        self.roiList = QListWidget()
        self.roiList.setSelectionMode(QAbstractItemView.ExtendedSelection)

        lay.setContentsMargins(12, 14, 12, 12)
        lay.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        def mkbtn(text, objname, h=36):
            b = QPushButton(text)
            b.setObjectName(objname)
            b.setFixedHeight(h)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return b

        toolsBox = QGroupBox("ROI Tools")
        t = QGridLayout(toolsBox)
        t.setContentsMargins(10, 12, 10, 10)
        t.setHorizontalSpacing(10)
        t.setVerticalSpacing(10)
    
        self.btnROIrect   = mkbtn("ROI Rectangle", "greenButton")
        self.btnROIlasso  = mkbtn("ROI Lasso", "greenButton")
        self.btnROIsave   = mkbtn("Save ROI", "greenButton")
        self.btnROIclear  = mkbtn("Clear ROI", "grayButton")
        self.btnROIoverlay  = mkbtn("Overlay Mask", "grayButton")
        self.btnROIspectrum = mkbtn("ROI Spectrum", "purpleButton")
    
        t.addWidget(self.btnROIrect,  0, 0)
        t.addWidget(self.btnROIlasso, 0, 1)
        t.addWidget(self.btnROIsave,  1, 0)
        t.addWidget(self.btnROIclear, 1, 1)
        t.addWidget(self.btnROIoverlay,  2, 0)
        t.addWidget(self.btnROIspectrum, 2, 1)

        lay.addWidget(toolsBox)

        lay.addWidget(QLabel("Saved ROIs:"))
        self.roiList.setToolTip(
                                "Hold Ctrl or Shift to select multiple ROIs"
                            )
        self.roiList.setMinimumHeight(70)
        lay.addWidget(self.roiList)
    
        lay.addWidget(QLabel("ROI Compare / Volcano Selection:"))
        rowAB = QHBoxLayout()
        rowAB.setSpacing(6)
    
        self.comboROI_A = QComboBox()
        self.comboROI_B = QComboBox()
        self.comboROI_A.setPlaceholderText("ROI A")
        self.comboROI_B.setPlaceholderText("ROI B")
        self._fix_combo_popup(self.comboROI_A)
        self._fix_combo_popup(self.comboROI_B)
    
        rowAB.addWidget(QLabel("ROI A:"))
        rowAB.addWidget(self.comboROI_A)
        rowAB.addWidget(QLabel("ROI B:"))
        rowAB.addWidget(self.comboROI_B)
        lay.addLayout(rowAB)
    
        self.lblROIst = QLabel("ROI: none")
        self.lblROIst.setStyleSheet("color:#374151; font-weight:900;")
        lay.addWidget(self.lblROIst)
        self.plotTypeCombo.clear()
        self.plotTypeCombo.addItems(["Box Plot", "Violin Plot"])    
    
        self.panelROIAnalizler = QGroupBox("ROI Statistics")
        p = QGridLayout(self.panelROIAnalizler)
        p.setContentsMargins(10, 12, 10, 10)
        p.setHorizontalSpacing(10)
        p.setVerticalSpacing(10)
        self.btnMultiROIStats = QPushButton("Multi-ROI Statistics")

        self.btnROIstats    = mkbtn("ROI Stats", "purpleButton")
        self.btnROISimilarity= mkbtn("ROI Similarity", "purpleButton")

        self.btnROIcompare  = mkbtn("ROI Compare (Box)", "DataButton", h=46)
        self.btnROIcmpV     = mkbtn("ROI Compare (Violin)", "DataButton", h=46)
        self.btnVolcano     = mkbtn("Volcano Plot", "DataButton", h=46)
        #self.btnROIcsv      = mkbtn("Export ROI CSV", "grayButton", h=46)
        self.btnMultiROIStats = mkbtn("Multi-ROI Statistics", "purpleButton", h=46)
    
        p.addWidget(self.btnROIstats,    0, 0)

        p.addWidget(self.btnROISimilarity,0,1)
        p.addWidget(self.btnROIcompare,  1, 0)
        p.addWidget(self.btnROIcmpV,     1, 1)
        p.addWidget(self.btnMultiROIStats, 2,0) 
        p.addWidget(self.btnVolcano,     2, 1)     
        
        #p.addWidget(self.btnROIcsv,      3, 0)
        
        self.btnROIcsv = QPushButton("Export ROI CSV")
        self.btnROIcsv.setObjectName("grayButton")
        self.btnROIcsv.setMinimumHeight(42)
        self.btnROIcsv.setEnabled(False)        
        lay.addWidget(self.btnROIcsv)  
        
        lay.addWidget(self.panelROIAnalizler)
        
def build_menu(self):
        self.menubar = QMenuBar(self)
        self.menubar.setNativeMenuBar(False)
        # File
        fileMenu = self.menubar.addMenu("File")
        self.loadAction = QAction("Load File", self)
        self.loadAction.setShortcut("Ctrl+L")
        fileMenu.addAction(self.loadAction)
        fileMenu.addSeparator()
        self.exitAction = QAction("Exit", self)
        fileMenu.addAction(self.exitAction)
        # Tools
        toolsMenu = self.menubar.addMenu("Tools")
        self.spectrumAction = QAction("Spectrum", self)
        self.imageAction = QAction("Update Images", self)
        self.plotAction = QAction("Update Plot", self)
        self.aiAction = QAction("AI Assist: Top Peak → Image 1", self)
        toolsMenu.addAction(self.spectrumAction)
        toolsMenu.addAction(self.imageAction)
        toolsMenu.addAction(self.plotAction)
        toolsMenu.addSeparator()
        toolsMenu.addAction(self.aiAction)
            
        helpMenu = self.menubar.addMenu("Help")
    
        self.aboutAction = QAction("About", self)
        helpMenu.addAction(self.aboutAction)
    
        helpMenu.addSeparator()
        self.benchExportAction = QAction("Export Benchmark CSV", self)
        helpMenu.addAction(self.benchExportAction)
        self.benchClearAction = QAction("Clear Benchmark Log", self)
        helpMenu.addAction(self.benchClearAction)
    
        self.setMenuBar(self.menubar)  
        



