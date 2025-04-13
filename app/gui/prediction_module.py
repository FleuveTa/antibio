from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTabWidget, QLabel)
from PyQt5.QtCore import Qt

from app.gui.data_tab import DataImportTab
from app.gui.preprocessing_tab import PreprocessingTab
from app.gui.feature_tab import FeatureEngineeringTab
from app.gui.model_tab import ModelTrainingTab
from app.gui.results_tab import ResultsVisualizationTab


class PredictionModuleWidget(QWidget):
    """
    Container widget for the Antibiotic Prediction module
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
        title_label = QLabel("Predict Antibiotic Residue")
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
        self.data_tab = DataImportTab(self.app_data)
        self.preprocess_tab = PreprocessingTab(self.app_data)  # New separate preprocessing tab
        self.feature_tab = FeatureEngineeringTab(self.app_data)
        self.model_tab = ModelTrainingTab(self.app_data)
        self.results_tab = ResultsVisualizationTab(self.app_data)

        # Add tabs to widget
        self.tabs.addTab(self.data_tab, "Data Import")
        self.tabs.addTab(self.preprocess_tab, "Preprocessing")
        self.tabs.addTab(self.feature_tab, "Feature Extraction")
        self.tabs.addTab(self.model_tab, "Model Training")
        self.tabs.addTab(self.results_tab, "Results & Analysis")

        # Connect signals
        self.data_tab.data_loaded.connect(self.enable_preprocess_tab)
        self.preprocess_tab.preprocessing_done.connect(self.enable_feature_tab)
        self.feature_tab.features_ready.connect(self.enable_model_tab)
        self.model_tab.model_ready.connect(self.enable_results_tab)

        # Initially disable tabs that require previous steps
        self.tabs.setTabEnabled(1, False)  # Preprocessing tab
        self.tabs.setTabEnabled(2, False)  # Feature tab
        self.tabs.setTabEnabled(3, False)  # Model tab
        self.tabs.setTabEnabled(4, False)  # Results tab

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def enable_preprocess_tab(self, enabled):
        """Enable the preprocessing tab when data is loaded"""
        self.tabs.setTabEnabled(1, enabled)
        if enabled:
            self.tabs.setCurrentIndex(1)

    def enable_feature_tab(self, enabled):
        """Enable the feature tab when preprocessing is done"""
        self.tabs.setTabEnabled(2, enabled)
        if enabled:
            self.tabs.setCurrentIndex(2)

    def enable_model_tab(self, enabled):
        """Enable the model tab when features are ready"""
        # CRITICAL FIX: Ensure feature_matrix is properly passed to model tab
        if enabled and "feature_matrix" in self.app_data and self.app_data["feature_matrix"] is not None:
            print("\n==== ENABLING MODEL TAB ====")
            print(f"Feature matrix exists with shape: {self.app_data['feature_matrix'].shape}")
            print(f"Feature matrix columns: {self.app_data['feature_matrix'].columns.tolist()}")

            # Ensure metadata columns are stored separately
            metadata_cols = [col for col in self.app_data['feature_matrix'].columns
                            if col in ['concentration', 'antibiotic']]
            if metadata_cols:
                print(f"Storing metadata columns in app_data: {metadata_cols}")
                self.app_data["metadata_columns"] = metadata_cols
            print("===========================\n")

        self.tabs.setTabEnabled(3, enabled)
        if enabled:
            self.tabs.setCurrentIndex(3)

    def enable_results_tab(self, enabled):
        """Enable the results tab when model is ready"""
        self.tabs.setTabEnabled(4, enabled)
        if enabled:
            self.tabs.setCurrentIndex(4)
