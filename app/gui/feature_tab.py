from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QSplitter, QHeaderView, QMessageBox)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal
from datetime import datetime

from app.core.feature_eng import (extract_voltammetric_features, extract_features_from_samples,
                                 VOLTAMMETRIC_FEATURE_DESCRIPTIONS)
from app.core.experiment_manager import ExperimentManager


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
        extraction_group = QGroupBox("Electrochemical Feature Extraction")
        extraction_layout = QVBoxLayout()

        # Introduction text
        intro_label = QLabel(
            "<p>Extract meaningful features from your voltammetric data to use in predictive models.</p>"
            "<p>Features are organized into categories that are relevant for electrochemical analysis.</p>"
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet("font-size: 11pt; margin-bottom: 10px;")
        extraction_layout.addWidget(intro_label)

        # Feature category selection
        category_layout = QHBoxLayout()

        # Category selection
        category_layout.addWidget(QLabel("Feature Categories:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems([
            "All Categories",
            "Basic Features",
            "Peak Features",
            "Shape Features",
            "Derivative Features",
            "Area Features"
        ])
        self.category_combo.currentIndexChanged.connect(self.update_feature_display)
        category_layout.addWidget(self.category_combo)

        # Extract button
        self.extract_btn = QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.extract_features)
        self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        category_layout.addWidget(self.extract_btn)

        extraction_layout.addLayout(category_layout)

        # Feature table with descriptions
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(["Category", "Feature", "Value", "Description"])
        self.feature_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.feature_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.feature_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.feature_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.feature_table.setAlternatingRowColors(True)
        self.feature_table.setStyleSheet("alternate-background-color: #f0f0f0;")
        extraction_layout.addWidget(self.feature_table)

        # Save features button
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Features")
        self.save_btn.clicked.connect(self.save_features)
        self.save_btn.setEnabled(False)  # Initially disabled until features are extracted
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        save_layout.addWidget(self.save_btn)

        # Continue to model button
        self.continue_btn = QPushButton("Continue to Model Training →")
        self.continue_btn.clicked.connect(self.continue_to_model)
        self.continue_btn.setEnabled(False)  # Initially disabled until features are saved
        self.continue_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        save_layout.addWidget(self.continue_btn)

        extraction_layout.addLayout(save_layout)
        extraction_group.setLayout(extraction_layout)
        splitter.addWidget(extraction_group)

        # Feature Visualization Section
        viz_group = QGroupBox("Feature Visualization")
        viz_layout = QVBoxLayout()

        # Add a label explaining the visualization
        viz_info = QLabel(
            "<p>This section will show visualizations of your extracted features.</p>"
            "<p>In the future, this will include:</p>"
            "<ul>"
            "<li>Feature importance charts</li>"
            "<li>Feature correlation heatmaps</li>"
            "<li>Feature distribution plots</li>"
            "</ul>"
        )
        viz_info.setWordWrap(True)
        viz_info.setStyleSheet("font-size: 11pt; margin: 10px;")
        viz_layout.addWidget(viz_info)

        viz_group.setLayout(viz_layout)
        splitter.addWidget(viz_group)

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Initialize with disabled state
        self.set_enabled(False)

    def set_enabled(self, enabled):
        """Enable or disable controls based on data availability"""
        self.extract_btn.setEnabled(enabled)

    def update_column_list(self):
        """Update UI based on data availability"""
        if self.app_data.get("dataset") is not None:
            self.set_enabled(True)

    def extract_features(self):
        """Extract electrochemical features from the voltammetric data"""
        if "dataset" not in self.app_data or self.app_data["dataset"] is None:
            QMessageBox.warning(self, "No Data", "Please load data before extracting features.")
            return

        data = self.app_data["dataset"]

        try:
            # Check data format
            if "Potential" in data.columns and "Current" in data.columns:
                # Standard format with Potential and Current columns
                QMessageBox.information(self, "Processing", "Processing data in standard format (Potential/Current columns)...")
                features = extract_voltammetric_features(data)
                self.app_data["features"] = features
                self.app_data["feature_matrix"] = None  # No feature matrix for this format
            else:
                # New format with voltage columns and metadata
                QMessageBox.information(self, "Processing", "Processing data in new format (voltage columns with metadata)...")
                feature_matrix = extract_features_from_samples(data)

                if feature_matrix.empty:
                    QMessageBox.warning(self, "Error", "Could not extract features from the data. Please check the data format.")
                    return

                # Store both the feature matrix and a flattened version for the UI
                self.app_data["feature_matrix"] = feature_matrix

                # Debug print to verify feature matrix
                print("\n==== FEATURE EXTRACTION COMPLETE ====")
                print(f"Feature matrix shape: {feature_matrix.shape}")
                print(f"Feature matrix columns: {feature_matrix.columns.tolist()}")
                metadata_cols = [col for col in feature_matrix.columns if col in ['concentration', 'antibiotic']]
                print(f"Metadata columns: {metadata_cols}")
                print("===================================\n")

                # For backward compatibility, create a flattened version of the first sample's features
                # This is used for the feature display in the UI
                first_sample_features = {}
                for col in feature_matrix.columns:
                    if col not in ['concentration', 'antibiotic', 'path']:
                        first_sample_features[col] = feature_matrix[col].iloc[0]

                self.app_data["features"] = first_sample_features

                # DIRECT APPROACH: Also store the metadata columns separately for easy access
                self.app_data["metadata_columns"] = metadata_cols

                # Show summary of extracted features
                num_samples = len(feature_matrix)
                num_features = len(feature_matrix.columns) - sum(1 for col in feature_matrix.columns if col in ['concentration', 'antibiotic', 'path'])

                QMessageBox.information(
                    self,
                    "Feature Extraction Complete",
                    f"Successfully extracted {num_features} features from {num_samples} samples.\n\n"
                    f"Metadata columns preserved: {[col for col in feature_matrix.columns if col in ['concentration', 'antibiotic']]}\n\n"
                    f"The feature matrix is now ready for model training."
                )

            # Update feature display
            self.update_feature_display()

            # Enable save button
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error extracting features: {str(e)}")

    def update_feature_display(self):
        """Update the feature table based on the selected category"""
        if "features" not in self.app_data or not self.app_data["features"]:
            return

        features = self.app_data["features"]
        selected_category = self.category_combo.currentText()

        # Filter features by category
        if selected_category == "All Categories":
            filtered_features = features
        else:
            # Convert "Basic Features" to "basic" for filtering
            category_prefix = selected_category.lower().split()[0]
            filtered_features = {k: v for k, v in features.items() if k.startswith(category_prefix)}

        # Update feature table
        self.feature_table.setRowCount(len(filtered_features))

        for i, (key, value) in enumerate(filtered_features.items()):
            # Split the key to get category and feature name
            parts = key.split('_', 1)
            category = parts[0]
            feature_name = parts[1] if len(parts) > 1 else key

            # Format category name for display
            display_category = category.capitalize()

            # Get feature description
            description = VOLTAMMETRIC_FEATURE_DESCRIPTIONS.get(key, "")

            # Add items to table
            self.feature_table.setItem(i, 0, QTableWidgetItem(display_category))
            self.feature_table.setItem(i, 1, QTableWidgetItem(feature_name))

            # Format value based on its magnitude
            if abs(value) < 0.001 or abs(value) > 1000:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.6f}"

            self.feature_table.setItem(i, 2, QTableWidgetItem(formatted_value))
            self.feature_table.setItem(i, 3, QTableWidgetItem(description))

            # Color-code rows by category
            if category == "basic":
                color = "#e3f2fd"  # Light blue
            elif category == "peak":
                color = "#e8f5e9"  # Light green
            elif category == "shape":
                color = "#fff3e0"  # Light orange
            elif category == "derivative":
                color = "#f3e5f5"  # Light purple
            elif category == "area":
                color = "#e0f7fa"  # Light cyan
            else:
                color = "#ffffff"  # White

            for j in range(4):
                item = self.feature_table.item(i, j)
                if item:
                    item.setBackground(QColor(color))

        # Resize rows to content
        self.feature_table.resizeRowsToContents()

    def save_features(self):
        """Save extracted features to the experiment"""
        if "features" not in self.app_data or not self.app_data["features"]:
            QMessageBox.warning(self, "No Features", "No features to save. Please extract features first.")
            return

        if "current_experiment_id" not in self.app_data or not self.app_data["current_experiment_id"]:
            QMessageBox.warning(self, "No Experiment", "No active experiment. Please load or create an experiment first.")
            return

        try:
            # Get the experiment manager
            experiment_manager = ExperimentManager()

            # Get the current experiment ID
            experiment_id = self.app_data["current_experiment_id"]

            # Check if we have a feature matrix (new format) or just features (old format)
            if "feature_matrix" in self.app_data and self.app_data["feature_matrix"] is not None:
                # New format - save the feature matrix
                feature_matrix = self.app_data["feature_matrix"]

                # Create metadata
                metadata = {
                    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sample_count": len(feature_matrix),
                    "feature_count": len(feature_matrix.columns) - sum(1 for col in feature_matrix.columns if col in ['concentration', 'antibiotic', 'path']),
                    "metadata_columns": [col for col in feature_matrix.columns if col in ['concentration', 'antibiotic']],
                    "description": "Sample-by-sample electrochemical features extracted from voltammetric data"
                }

                # Save the feature matrix
                file_path = experiment_manager.save_feature_matrix(experiment_id, feature_matrix, metadata)

                # Store the features file path in app_data
                self.app_data["features_file_path"] = file_path

                # Enable the continue button
                self.continue_btn.setEnabled(True)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Feature matrix with {len(feature_matrix)} samples saved successfully to:\n{file_path}"
                )
            else:
                # Old format - save the features dictionary
                features = self.app_data["features"]

                # Create metadata
                metadata = {
                    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_count": len(features),
                    "categories": list(set([k.split('_')[0] for k in features.keys()])),
                    "description": "Electrochemical features extracted from voltammetric data"
                }

                # Save the features
                file_path = experiment_manager.save_features(experiment_id, features, metadata)

                # Store the features file path in app_data
                self.app_data["features_file_path"] = file_path

                # Enable the continue button
                self.continue_btn.setEnabled(True)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Features saved successfully to:\n{file_path}"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving features: {str(e)}")

        finally:
            # Close the experiment manager
            if 'experiment_manager' in locals():
                experiment_manager.close()

    def continue_to_model(self):
        """Signal that features are ready and continue to model training"""
        # DIRECT APPROACH: Verify that feature_matrix is in app_data before continuing
        print("\n==== CONTINUE TO MODEL CALLED ====")
        print(f"app_data keys: {list(self.app_data.keys())}")

        if "feature_matrix" in self.app_data and self.app_data["feature_matrix"] is not None:
            feature_matrix = self.app_data["feature_matrix"]
            print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Feature matrix columns: {feature_matrix.columns.tolist()}")

            # CRITICAL FIX: Make sure the feature_matrix is properly stored in app_data
            # This ensures it's accessible to other tabs
            self.app_data["feature_matrix"] = feature_matrix

            metadata_cols = [col for col in feature_matrix.columns if col in ['concentration', 'antibiotic']]
            print(f"Metadata columns: {metadata_cols}")

            # CRITICAL FIX: Store metadata columns separately for easy access
            self.app_data["metadata_columns"] = metadata_cols

            if metadata_cols:
                # Show a message about available target variables
                QMessageBox.information(
                    self,
                    "Target Variables Available",
                    f"The following columns can be used as target variables in the Model Training tab:\n\n{', '.join(metadata_cols)}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "No Target Variables",
                    "No metadata columns (concentration, antibiotic) found in the dataset. You may need to add these columns to your data."
                )
        else:
            print("No feature_matrix found in app_data!")

        print("===================================\n")

        # Signal to continue to model training tab
        self.features_ready.emit(True)
        QMessageBox.information(self, "Continuing", "Proceeding to Model Training tab.")

    def select_features(self):
        """This method is kept for backward compatibility"""
        # Signal that features are ready
        self.features_ready.emit(True)

    # Visualization methods have been removed as part of simplification

    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)
        # Update column list and enable/disable controls based on data
        self.update_column_list()
        self.set_enabled(self.app_data.get("dataset") is not None)