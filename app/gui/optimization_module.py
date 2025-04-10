from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTabWidget, QLabel)
from PyQt5.QtCore import Qt

from app.gui.parameter_data_tab import ParameterDataImportTab
from app.gui.regression_model_tab import RegressionModelTab
from app.gui.optimization_results_tab import OptimizationResultsTab


class ParameterOptimizationWidget(QWidget):
    """
    Container widget for the Parameter Optimization module
    """
    def __init__(self, app_data, parent=None):
        super().__init__(parent)
        self.app_data = app_data
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header with navigation and title
        header_layout = QHBoxLayout()
        
        # Back button
        self.back_btn = QPushButton("← Back to Home")
        self.back_btn.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.back_btn)
        
        # Title
        title_label = QLabel("Find Optimal Device Parameters")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Empty widget for symmetry
        spacer = QPushButton("")
        spacer.setEnabled(False)
        spacer.setStyleSheet("background-color: transparent; border: none;")
        spacer.setFixedWidth(self.back_btn.sizeHint().width())
        header_layout.addWidget(spacer)
        
        layout.addLayout(header_layout)
        
        # Create tab widget for workflow
        self.tabs = QTabWidget()
        
        # Create individual tabs
        self.data_tab = ParameterDataImportTab(self.app_data)
        self.model_tab = RegressionModelTab(self.app_data)
        self.results_tab = OptimizationResultsTab(self.app_data)
        
        # Add tabs to widget
        self.tabs.addTab(self.data_tab, "Data Import")
        self.tabs.addTab(self.model_tab, "Regression Model")
        self.tabs.addTab(self.results_tab, "Optimization Results")
        
        # Connect signals
        self.data_tab.data_loaded.connect(self.enable_model_tab)
        self.model_tab.model_fitted.connect(self.enable_results_tab)
        
        # Initially disable tabs that require previous steps
        self.tabs.setTabEnabled(1, False)  # Model tab
        self.tabs.setTabEnabled(2, False)  # Results tab
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def enable_model_tab(self, enabled):
        """Enable the model tab when data is loaded"""
        self.tabs.setTabEnabled(1, enabled)
        if enabled:
            self.tabs.setCurrentIndex(1)
    
    def enable_results_tab(self, enabled):
        """Enable the results tab when model is fitted"""
        self.tabs.setTabEnabled(2, enabled)
        if enabled:
            self.tabs.setCurrentIndex(2)
