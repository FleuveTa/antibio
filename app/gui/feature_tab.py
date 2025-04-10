from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QGridLayout, QCheckBox, QSpinBox,
                             QRadioButton, QButtonGroup, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.core.feature_eng import (extract_voltammetric_features, 
                                 extract_time_series_features,
                                 select_features)


class FeatureEngineeringTab(QWidget):
    features_ready = pyqtSignal(bool)
    
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # Feature Extraction Section
        extraction_group = QGroupBox("Feature Extraction")
        extraction_layout = QVBoxLayout()
        
        # Feature type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Feature Type:"))
        self.feature_type_combo = QComboBox()
        self.feature_type_combo.addItems([
            "Electrochemical Features", 
            "Statistical Features",
            "Time Domain Features", 
            "Frequency Domain Features",
            "All Features"
        ])
        type_layout.addWidget(self.feature_type_combo)
        
        # Target column selection (for time series data)
        type_layout.addWidget(QLabel("Target Column:"))
        self.target_column_combo = QComboBox()
        type_layout.addWidget(self.target_column_combo)
        
        # Extract button
        self.extract_btn = QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.extract_features)
        type_layout.addWidget(self.extract_btn)
        
        extraction_layout.addLayout(type_layout)
        
        # Feature table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(3)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value", "Include"])
        self.feature_table.horizontalHeader().setStretchLastSection(True)
        extraction_layout.addWidget(self.feature_table)
        
        # Advanced options
        options_layout = QGridLayout()
        
        # Windowing options for time series
        options_layout.addWidget(QLabel("Window Size:"), 0, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 1000)
        self.window_spin.setValue(100)
        options_layout.addWidget(self.window_spin, 0, 1)
        
        options_layout.addWidget(QLabel("Overlap:"), 0, 2)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 99)
        self.overlap_spin.setValue(50)
        self.overlap_spin.setSuffix("%")
        options_layout.addWidget(self.overlap_spin, 0, 3)
        
        # Peak detection threshold for voltammetric data
        options_layout.addWidget(QLabel("Peak Threshold:"), 1, 0)
        self.peak_spin = QSpinBox()
        self.peak_spin.setRange(1, 100)
        self.peak_spin.setValue(10)
        self.peak_spin.setSuffix("%")
        options_layout.addWidget(self.peak_spin, 1, 1)
        
        # Custom feature option
        self.custom_check = QCheckBox("Add custom calculation")
        options_layout.addWidget(self.custom_check, 1, 2, 1, 2)
        
        extraction_layout.addLayout(options_layout)
        extraction_group.setLayout(extraction_layout)
        splitter.addWidget(extraction_group)
        
        # Feature Selection Section
        selection_group = QGroupBox("Feature Selection")
        selection_layout = QVBoxLayout()
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Selection Method:"))
        
        self.method_group = QButtonGroup()
        
        self.correlation_radio = QRadioButton("Correlation")
        self.correlation_radio.setChecked(True)
        self.method_group.addButton(self.correlation_radio)
        method_layout.addWidget(self.correlation_radio)
        
        self.variance_radio = QRadioButton("Variance")
        self.method_group.addButton(self.variance_radio)
        method_layout.addWidget(self.variance_radio)
        
        self.rfe_radio = QRadioButton("RFE")
        self.method_group.addButton(self.rfe_radio)
        method_layout.addWidget(self.rfe_radio)
        
        self.pca_radio = QRadioButton("PCA")
        self.method_group.addButton(self.pca_radio)
        method_layout.addWidget(self.pca_radio)
        
        # Threshold/number of features
        method_layout.addWidget(QLabel("Threshold/Num Features:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 100)
        self.threshold_spin.setValue(20)
        method_layout.addWidget(self.threshold_spin)
        
        # Select button
        self.select_btn = QPushButton("Select Features")
        self.select_btn.clicked.connect(self.select_features)
        method_layout.addWidget(self.select_btn)
        
        selection_layout.addLayout(method_layout)
        
        # Feature visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        selection_layout.addWidget(self.canvas)
        
        # Visualization type
        viz_layout = QHBoxLayout()
        self.importance_btn = QPushButton("Feature Importance")
        self.importance_btn.clicked.connect(lambda: self.visualize_features("importance"))
        viz_layout.addWidget(self.importance_btn)
        
        self.correlation_btn = QPushButton("Correlation Matrix")
        self.correlation_btn.clicked.connect(lambda: self.visualize_features("correlation"))
        viz_layout.addWidget(self.correlation_btn)
        
        self.pca_btn = QPushButton("PCA Plot")
        self.pca_btn.clicked.connect(lambda: self.visualize_features("pca"))
        viz_layout.addWidget(self.pca_btn)
        
        selection_layout.addLayout(viz_layout)
        
        # Selected features summary
        selection_layout.addWidget(QLabel("Selected Features:"))
        self.selected_features_label = QLabel("No features selected")
        selection_layout.addWidget(self.selected_features_label)
        
        # Confirm button
        self.confirm_btn = QPushButton("Confirm Features for Model Training")
        self.confirm_btn.clicked.connect(self.confirm_features)
        selection_layout.addWidget(self.confirm_btn)
        
        selection_group.setLayout(selection_layout)
        splitter.addWidget(selection_group)
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
        # Initialize with disabled state
        self.set_enabled(False)
        
    def set_enabled(self, enabled):
        """Enable or disable controls based on data availability"""
        self.extract_btn.setEnabled(enabled)
        self.select_btn.setEnabled(enabled)
        self.importance_btn.setEnabled(enabled)
        self.correlation_btn.setEnabled(enabled)
        self.pca_btn.setEnabled(enabled)
        self.confirm_btn.setEnabled(enabled)
    
    def update_column_list(self):
        """Update the list of available columns from the dataset"""
        if self.app_data.get("processed_data") is not None:
            data = self.app_data["processed_data"]
            self.target_column_combo.clear()
            self.target_column_combo.addItems(data.columns)
    
    def extract_features(self):
        """Extract features based on user selection"""
        if self.app_data.get("processed_data") is None:
            return
            
        data = self.app_data["processed_data"]
        sensor_type = self.app_data.get("sensor_type", "")
        feature_type = self.feature_type_combo.currentText()
        
        features = {}
        
        # Extract different types of features based on selection
        if feature_type in ["Electrochemical Features", "All Features"]:
            if sensor_type == "Voltammetric":
                volt_features = extract_voltammetric_features(data)
                features.update(volt_features)
            
        if feature_type in ["Statistical Features", "All Features"]:
            # Extract basic statistical features for all numeric columns
            for col in data.select_dtypes(include=np.number).columns:
                features[f"{col}_mean"] = data[col].mean()
                features[f"{col}_std"] = data[col].std()
                features[f"{col}_min"] = data[col].min()
                features[f"{col}_max"] = data[col].max()
                features[f"{col}_range"] = data[col].max() - data[col].min()
        
        if feature_type in ["Time Domain Features", "All Features"]:
            target_col = self.target_column_combo.currentText()
            if target_col:
                time_features = extract_time_series_features(data, target_col)
                # Prefix with column name
                time_features = {f"{target_col}_{k}": v for k, v in time_features.items()}
                features.update(time_features)
        
        # TODO: Add frequency domain features
        
        # Update feature table
        self.feature_table.setRowCount(len(features))
        for i, (key, value) in enumerate(features.items()):
            self.feature_table.setItem(i, 0, QTableWidgetItem(key))
            self.feature_table.setItem(i, 1, QTableWidgetItem(str(round(value, 6))))
            
            # Add checkbox for feature selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.feature_table.setCellWidget(i, 2, checkbox)
        
        # Store features in app_data
        self.app_data["features"] = features
        
        # Update feature visualization
        self.visualize_features("importance")
    
    def select_features(self):
        """Select features based on the chosen method"""
        if self.app_data.get("features") is None:
            return
            
        # Get selected features from checkboxes
        selected_features = {}
        for i in range(self.feature_table.rowCount()):
            feature_name = self.feature_table.item(i, 0).text()
            checkbox = self.feature_table.cellWidget(i, 2)
            if checkbox.isChecked():
                value_str = self.feature_table.item(i, 1).text()
                selected_features[feature_name] = float(value_str)
        
        # Convert to DataFrame for selection methods
        feature_df = pd.DataFrame([selected_features])
        
        # Get selection method
        if self.correlation_radio.isChecked():
            method = "correlation"
        elif self.variance_radio.isChecked():
            method = "variance"
        elif self.rfe_radio.isChecked():
            method = "rfe"
        elif self.pca_radio.isChecked():
            method = "pca"
        
        # Apply feature selection if we have a target
        if self.app_data.get("target") is not None:
            target = self.app_data["target"]
            threshold = self.threshold_spin.value() / 100  # Convert percentage to fraction
            
            # Note: In a real app, you would connect to your select_features implementation
            # This is simplified here
            selected_feature_names = list(selected_features.keys())[:self.threshold_spin.value()]
            
            # Update selected features
            self.app_data["selected_features"] = selected_feature_names
            
            # Update label
            self.selected_features_label.setText(", ".join(selected_feature_names))
        else:
            # If no target, just use the top N features
            top_n = min(self.threshold_spin.value(), len(selected_features))
            selected_feature_names = list(selected_features.keys())[:top_n]
            
            # Update selected features
            self.app_data["selected_features"] = selected_feature_names
            
            # Update label
            self.selected_features_label.setText(", ".join(selected_feature_names))
    
    def visualize_features(self, viz_type):
        """Visualize features based on the selected visualization type"""
        if self.app_data.get("features") is None:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        features = self.app_data["features"]
        
        if viz_type == "importance":
            # Plot feature values as a bar chart
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            # Sort by value for better visualization
            sorted_indices = np.argsort(feature_values)
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_values = [feature_values[i] for i in sorted_indices]
            
            # Plot only top 15 if we have more
            if len(sorted_names) > 15:
                sorted_names = sorted_names[-15:]
                sorted_values = sorted_values[-15:]
                
            ax.barh(sorted_names, sorted_values)
            ax.set_title("Feature Values")
            ax.set_xlabel("Value")
            
        elif viz_type == "correlation":
            # Create a correlation matrix visualization
            # This is a simplified version - in a real app, you would compute
            # actual correlations between features
            
            # Create a mock correlation matrix for demonstration
            feature_names = list(features.keys())[:10]  # Limit to 10 for readability
            n_features = len(feature_names)
            
            # Random correlation matrix for demo purposes
            np.random.seed(42)
            corr_matrix = np.random.rand(n_features, n_features)
            # Make it symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(corr_matrix, 1)
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            self.figure.colorbar(im, ax=ax)
            
            # Set ticks
            ax.set_xticks(np.arange(n_features))
            ax.set_yticks(np.arange(n_features))
            ax.set_xticklabels(feature_names, rotation=90)
            ax.set_yticklabels(feature_names)
            
            ax.set_title("Feature Correlation Matrix")
            
        elif viz_type == "pca":
            # Create a mock PCA plot for demonstration
            ax.set_title("PCA of Features")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            
            # Random scatter points for demo
            np.random.seed(42)
            x = np.random.randn(20)
            y = np.random.randn(20)
            
            # Different colors for different classes
            colors = ['r', 'g', 'b', 'y']
            classes = np.random.randint(0, 4, 20)
            
            for i, color in enumerate(colors):
                mask = classes == i
                ax.scatter(x[mask], y[mask], c=color, label=f'Class {i}')
            
            ax.legend()
        
        self.canvas.draw()
    
    def confirm_features(self):
        """Confirm feature selection and signal ready for model training"""
        if self.app_data.get("selected_features") is not None:
            self.features_ready.emit(True)
    
    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)
        # Update column list and enable/disable controls based on data
        self.update_column_list()
        self.set_enabled(self.app_data.get("processed_data") is not None)