from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QSplitter, QHeaderView, QMessageBox)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal
from datetime import datetime
import pandas as pd

from app.core.feature_eng import apply_pca_to_wide_data
from app.core.preprocessing import preprocess_wide_data
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
            "PCA Features"
        ])
        self.category_combo.currentIndexChanged.connect(self.update_feature_display)
        category_layout.addWidget(self.category_combo)

        extraction_layout.addLayout(category_layout)

        # Feature extraction method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Feature Extraction Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "PCA Features"  # Currently only PCA is available
        ])
        self.method_combo.setToolTip("'PCA Features' applies Principal Component Analysis to reduce dimensionality of raw voltage values.\nMore methods will be added in future updates.")
        method_layout.addWidget(self.method_combo)

        # Extract button
        self.extract_btn = QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.extract_features)
        self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        method_layout.addWidget(self.extract_btn)

        extraction_layout.addLayout(method_layout)

        # Feature table - simplified to only show feature names
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(1)
        self.feature_table.setHorizontalHeaderLabels(["Feature Name"])
        self.feature_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
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
        # Check if we have original_wide_data
        if self.app_data.get("original_wide_data") is not None:
            print("Found original_wide_data for feature extraction")
            self.set_enabled(True)

    def extract_features(self):
        """Extract electrochemical features from the voltammetric data"""
        # Check if we have original wide data to process
        print("\n==== CHECKING AVAILABLE DATA ====")
        print(f"app_data keys: {list(self.app_data.keys())}")

        if "original_wide_data" in self.app_data and self.app_data["original_wide_data"] is not None:
            # Use the original wide format data directly
            data = self.app_data["original_wide_data"]
            print(f"Found original_wide_data with shape: {data.shape}")
            print(f"original_wide_data columns: {data.columns.tolist()}")
            print(f"Using original_wide_data for feature extraction")
        else:
            print("No original wide data found in app_data")
            QMessageBox.warning(self, "No Data", "Please load data before extracting features.")
            return
        print("===================================\n")

        try:

            # Process data in wide format
            QMessageBox.information(self, "Processing", "Processing data in wide format...")

            # Currently only PCA is available as a feature extraction method
            print("\n==== FEATURE EXTRACTION METHOD ====\n")
            print("Using PCA for feature extraction")
            print("===================================\n")

            # Apply PCA to raw voltage values
            QMessageBox.information(self, "PCA Features", "Applying Principal Component Analysis to reduce dimensionality of raw voltage values.")
            print("Calling apply_pca_to_wide_data()")

            # First, preprocess the data if needed
            options = {
                "fill_missing": True,  # Fill missing values
                "smooth": True,  # Apply smoothing
                "baseline_correction": True  # Apply baseline correction
            }

            # Preprocess the data directly in wide format
            preprocessed_data, _ = preprocess_wide_data(data, options)
            QMessageBox.information(self, "Preprocessing", "Applied preprocessing to raw data: filled missing values, smoothing, and baseline correction.")

            # Apply PCA to preprocessed data
            # Use variance_threshold=0.95 to keep components that explain 95% of variance
            # Ensure we keep at least 3 components to capture more information
            feature_matrix, pca_model = apply_pca_to_wide_data(preprocessed_data, n_components=3, variance_threshold=0.95)

            # Store the PCA model for future use
            self.app_data["pca_model"] = pca_model

            print(f"PCA feature matrix shape: {feature_matrix.shape}")
            QMessageBox.information(self, "PCA Complete",
                                  f"PCA reduced dimensions from {len(preprocessed_data.columns)} to {len(feature_matrix.columns) - len([col for col in feature_matrix.columns if col in ['concentration', 'antibiotic', 'row_index']])} components.")

            if feature_matrix.empty:
                QMessageBox.warning(self, "Error", "Could not extract features from the data. Please check the data format.")
                return

            # Store both the feature matrix and a flattened version for the UI
            self.app_data["feature_matrix"] = feature_matrix

            # Debug print to verify feature matrix
            print("\n==== FEATURE EXTRACTION COMPLETE ====")
            print(f"Feature matrix shape: {feature_matrix.shape}")
            print(f"Feature matrix columns: {feature_matrix.columns.tolist()}")

            # Use a more flexible approach to identify metadata columns
            important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
            metadata_cols = []

            # First check for exact matches
            for col in feature_matrix.columns:
                if col in important_metadata:
                    metadata_cols.append(col)

            # Then check for case-insensitive matches if we didn't find any
            if not metadata_cols:
                for col in feature_matrix.columns:
                    if col.lower() in [meta.lower() for meta in important_metadata]:
                        metadata_cols.append(col)

            # If still no metadata columns, look for any non-feature columns
            if not metadata_cols:
                for col in feature_matrix.columns:
                    if not any(col.startswith(prefix) for prefix in ['basic_', 'peak_', 'shape_', 'derivative_', 'area_']):
                        if col != 'path' and col != 'row_index':  # Skip path and row_index columns
                            metadata_cols.append(col)

            print(f"Metadata columns: {metadata_cols}")
            print("===================================\n")

            # Store all samples' features for display in the UI
            # We'll create a dictionary where keys are sample indices and values are feature dictionaries
            all_samples_features = {}

            # For each sample (row) in the feature matrix
            for idx in range(len(feature_matrix)):
                sample_features = {}
                # Extract feature values for this sample
                for col in feature_matrix.columns:
                    if col not in metadata_cols and col != 'path' and col != 'row_index':
                        sample_features[col] = feature_matrix[col].iloc[idx]

                # Add metadata for this sample
                sample_metadata = {}
                for col in metadata_cols:
                    if col in feature_matrix.columns:
                        sample_metadata[col] = feature_matrix[col].iloc[idx]

                # Store features with metadata
                all_samples_features[idx] = {
                    'features': sample_features,
                    'metadata': sample_metadata
                }

            # Store all samples' features
            self.app_data["all_samples_features"] = all_samples_features

            # For backward compatibility, use the first sample's features for the UI
            if all_samples_features and 0 in all_samples_features:
                self.app_data["features"] = all_samples_features[0]['features']
            else:
                # Fallback to empty dictionary if no samples
                self.app_data["features"] = {}

            # DIRECT APPROACH: Also store the metadata columns separately for easy access
            self.app_data["metadata_columns"] = metadata_cols

            # Show summary of extracted features
            num_samples = len(feature_matrix)
            num_features = len(feature_matrix.columns) - len(metadata_cols) - sum(1 for col in feature_matrix.columns if col in ['path', 'row_index'])

            # Create a more detailed message about feature extraction
            message = f"Successfully extracted {num_features} features from {num_samples} samples.\n\n"
            message += f"Metadata columns preserved: {metadata_cols}\n\n"

            # Add information about PCA feature extraction
            message += "Principal Component Analysis (PCA) was applied to reduce dimensionality.\n"
            if "pca_model" in self.app_data:
                pca_model = self.app_data["pca_model"]
                message += f"PCA reduced dimensions to {pca_model.n_components_} components, explaining {pca_model.explained_variance_ratio_.sum()*100:.2f}% of variance.\n\n"

                # Add information about top components
                message += "Top 5 principal components and their explained variance:\n"
                for i in range(min(5, pca_model.n_components_)):
                    message += f"PC{i+1}: {pca_model.explained_variance_ratio_[i]*100:.2f}% of variance\n"
                message += "\n"
            else:
                message += "PCA was applied to extract the most important components.\n\n"

            message += "The feature matrix is now ready for model training.\n\n"
            message += f"These metadata columns can be used as target variables in the Model Training tab."

            QMessageBox.information(
                self,
                "Feature Extraction Complete",
                message
            )

            # Update feature display
            self.update_feature_display()

            # Enable save button
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error extracting features: {str(e)}")

    def update_feature_display(self):
        """Update the feature table based on the selected category"""
        # Check if we have features to display
        if "features" not in self.app_data or not self.app_data["features"]:
            return

        # Add a sample selector dropdown if we have multiple samples
        if "all_samples_features" in self.app_data and len(self.app_data["all_samples_features"]) > 1:
            # Check if we already have a sample selector
            if not hasattr(self, "sample_selector"):
                # Create a layout for the sample selector
                sample_selector_layout = QHBoxLayout()

                # Add a label
                sample_label = QLabel("Select Sample:")
                sample_selector_layout.addWidget(sample_label)

                # Create the sample selector dropdown
                self.sample_selector = QComboBox()

                # Add sample indices to the dropdown
                all_samples = self.app_data["all_samples_features"]
                for idx in all_samples.keys():
                    # Add metadata to the display text if available
                    display_text = f"Sample {idx+1}"
                    if 'metadata' in all_samples[idx]:
                        metadata = all_samples[idx]['metadata']
                        if metadata:
                            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                            display_text += f" ({metadata_str})"

                    self.sample_selector.addItem(display_text, idx)

                # Connect the change event
                self.sample_selector.currentIndexChanged.connect(self.on_sample_changed)

                # Add to layout
                sample_selector_layout.addWidget(self.sample_selector)

                # Add the sample selector layout to the top of the extraction layout
                # First, get the extraction group's layout
                extraction_layout = None
                for i in range(self.layout().count()):
                    widget = self.layout().itemAt(i).widget()
                    if isinstance(widget, QGroupBox) and widget.title() == "Electrochemical Feature Extraction":
                        extraction_layout = widget.layout()
                        break

                # If we found the extraction layout, insert the sample selector at the top
                if extraction_layout:
                    extraction_layout.insertLayout(0, sample_selector_layout)
                else:
                    # Fallback: just add it to the main layout
                    self.layout().insertLayout(0, sample_selector_layout)

            # Get the currently selected sample index
            current_idx = self.sample_selector.currentData()

            # If we have a valid index, use that sample's features
            if current_idx is not None and current_idx in self.app_data["all_samples_features"]:
                features = self.app_data["all_samples_features"][current_idx]['features']
            else:
                # Fallback to the first sample
                features = self.app_data["features"]
        else:
            # Just use the default features (first sample)
            features = self.app_data["features"]

        selected_category = self.category_combo.currentText()

        # For PCA features, we need to handle differently since they're in a different format
        # Check if we have a feature matrix with PC columns
        if "feature_matrix" in self.app_data and any(col.startswith("PC") for col in self.app_data["feature_matrix"].columns):
            # Create a dictionary of PCA features for display
            pca_features = {}
            feature_matrix = self.app_data["feature_matrix"]

            # Count PC columns for display
            pc_count = sum(1 for col in feature_matrix.columns if col.startswith("PC"))
            pca_features["Number of Principal Components"] = str(pc_count)

            # Add PCA model information if available
            if "pca_model" in self.app_data:
                pca_model = self.app_data["pca_model"]
                pca_features["Total Explained Variance"] = f"{pca_model.explained_variance_ratio_.sum()*100:.2f}%"

                # Add individual component variances
                for i, ratio in enumerate(pca_model.explained_variance_ratio_):
                    pca_features[f"PC{i+1} Explained Variance"] = f"{ratio*100:.2f}%"

            # Use PCA features instead of the original features
            filtered_features = pca_features
        else:
            # Filter features by category (for backward compatibility)
            if selected_category == "All Categories":
                filtered_features = features
            else:
                # Convert "Basic Features" to "basic" for filtering
                category_prefix = selected_category.lower().split()[0]
                filtered_features = {k: v for k, v in features.items() if k.startswith(category_prefix)}

        # Update feature table - simplified to only show feature names
        self.feature_table.setRowCount(len(filtered_features))

        # Sort features by name for better readability
        sorted_features = sorted(filtered_features.items())

        for i, (key, _) in enumerate(sorted_features):
            # Add feature name to table (just the key)
            self.feature_table.setItem(i, 0, QTableWidgetItem(key))

            # Color-code rows by category
            if "PC" in key or "Explained Variance" in key or "Principal Components" in key:
                color = "#e3f2fd"  # Light blue for PCA features
            else:
                color = "#ffffff"  # White for other features

            for j in range(4):
                item = self.feature_table.item(i, j)
                if item:
                    item.setBackground(QColor(color))

        # Resize rows to content
        self.feature_table.resizeRowsToContents()

    def on_sample_changed(self, _):
        """Handle when the user selects a different sample"""
        # Update the feature display with the newly selected sample
        self.update_feature_display()

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
        # Enable controls if original_wide_data is available
        has_data = self.app_data.get("original_wide_data") is not None
        self.set_enabled(has_data)