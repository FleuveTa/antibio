from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from PyQt5.QtCore import Qt

from app.gui.home_screen import HomeScreen
from app.gui.prediction_module import PredictionModuleWidget
from app.gui.optimization_module import ParameterOptimizationWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ElectroSense: Antibiotic Detection Tool")
        self.setMinimumSize(1024, 768)

        # Initialize shared data storage
        self.app_data = {
            "dataset": None,
            "features": None,
            "models": {},
            "results": {}
        }

        self.init_ui()

    def init_ui(self):
        # Create stacked widget for switching between screens
        self.stacked_widget = QStackedWidget()

        # Create home screen
        self.home_screen = HomeScreen()
        self.home_screen.module_selected.connect(self.switch_module)

        # Create module screens
        self.prediction_module = PredictionModuleWidget(self.app_data)
        self.prediction_module.back_btn.clicked.connect(self.show_home)

        self.optimization_module = ParameterOptimizationWidget(self.app_data)
        self.optimization_module.back_btn.clicked.connect(self.show_home)

        # Add widgets to stacked widget
        self.stacked_widget.addWidget(self.home_screen)  # index 0
        self.stacked_widget.addWidget(self.prediction_module)  # index 1
        self.stacked_widget.addWidget(self.optimization_module)  # index 2

        # Set central widget
        self.setCentralWidget(self.stacked_widget)

        # Set up status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def switch_module(self, module_name):
        """Switch to the selected module"""
        if module_name == "predict":
            self.stacked_widget.setCurrentIndex(1)  # Prediction module
        elif module_name == "optimize":
            self.stacked_widget.setCurrentIndex(2)  # Optimization module

    def show_home(self):
        """Show the home screen"""
        self.stacked_widget.setCurrentIndex(0)  # Home screen