from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal


class HomeScreen(QWidget):
    """
    Home screen that allows users to choose between the two main modules:
    1. Predict Antibiotic Residue
    2. Find Optimal Device Parameters
    """
    module_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("ElectroSense: Antibiotic Detection Tool")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Select one of the following modules:")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-size: 16px; margin: 20px 0;")
        layout.addWidget(desc_label)
        
        # Module buttons
        button_layout = QHBoxLayout()
        
        # Module 1: Antibiotic Prediction
        self.predict_btn = QPushButton("Predict Antibiotic Residue")
        self.predict_btn.setMinimumSize(200, 100)
        self.predict_btn.setStyleSheet("""
            font-size: 16px;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
        """)
        self.predict_btn.clicked.connect(lambda: self.module_selected.emit("predict"))
        button_layout.addWidget(self.predict_btn)
        
        # Add some spacing between buttons
        button_layout.addSpacing(20)
        
        # Module 2: Parameter Optimization
        self.optimize_btn = QPushButton("Find Optimal Device Parameters")
        self.optimize_btn.setMinimumSize(200, 100)
        self.optimize_btn.setStyleSheet("""
            font-size: 16px;
            padding: 10px;
            background-color: #2196F3;
            color: white;
        """)
        self.optimize_btn.clicked.connect(lambda: self.module_selected.emit("optimize"))
        button_layout.addWidget(self.optimize_btn)
        
        layout.addLayout(button_layout)
        
        # Add descriptions for each module
        descriptions_layout = QHBoxLayout()
        
        predict_desc = QLabel(
            "Train ML models to predict antibiotic residue\n"
            "using voltammetry data (CV and DPV)"
        )
        predict_desc.setAlignment(Qt.AlignCenter)
        predict_desc.setStyleSheet("margin-top: 10px;")
        descriptions_layout.addWidget(predict_desc)
        
        descriptions_layout.addSpacing(20)
        
        optimize_desc = QLabel(
            "Find optimal device parameters by fitting\n"
            "peak current data with regression models"
        )
        optimize_desc.setAlignment(Qt.AlignCenter)
        optimize_desc.setStyleSheet("margin-top: 10px;")
        descriptions_layout.addWidget(optimize_desc)
        
        layout.addLayout(descriptions_layout)
        
        # Add some spacing at the bottom
        layout.addStretch()
        
        self.setLayout(layout)
