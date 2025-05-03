from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
                             QTabWidget, QTableWidget, QTableWidgetItem, QSplitter,
                             QRadioButton, QButtonGroup, QSlider, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from app.core.models import AntibioticClassifier, RegressionClassifier


class TrainingThread(QThread):
    """Thread for running model training in background"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)

    def __init__(self, model_config, X, y, is_regression=False):
        super().__init__()
        self.model_config = model_config
        self.X = X
        self.y = y
        self.is_regression = is_regression

    def run(self):
        try:
            # Debug information
            print(f"\n==== TRAINING THREAD STARTED =====")
            print(f"Model config: {self.model_config}")
            print(f"X shape: {self.X.shape}")
            print(f"y shape: {self.y.shape}")
            print(f"y data type: {type(self.y[0]) if len(self.y) > 0 else 'empty'}")
            print(f"y first few values: {self.y[:5] if len(self.y) > 0 else 'empty'}")
            print(f"Is regression: {self.is_regression}")

            # Check for NaN values in X and y
            if np.isnan(self.X).any():
                nan_count = np.isnan(self.X).sum()
                print(f"Warning: Found {nan_count} NaN values in X. Replacing with 0.")
                self.X = np.nan_to_num(self.X, nan=0.0)

            # Create classifier with selected model type based on task type
            if self.is_regression:
                # Use regression model for concentration prediction
                classifier = RegressionClassifier(model_type=self.model_config['model_type'])
                print(f"Using RegressionClassifier for concentration prediction")

                # For regression, ensure y is numeric
                if not np.issubdtype(self.y.dtype, np.number):
                    print(f"Converting y to numeric for regression (current type: {self.y.dtype})")
                    try:
                        self.y = self.y.astype(float)
                    except Exception as e:
                        raise ValueError(f"Cannot convert target to numeric values: {e}")
            else:
                # Use classification model for antibiotic detection
                classifier = AntibioticClassifier(model_type=self.model_config['model_type'])
                print(f"Using AntibioticClassifier for classification")

            # Update progress
            self.progress_signal.emit(20)

            # Either optimize hyperparameters or just train
            if self.model_config.get('optimize', False):
                # For optimization, we might define a parameter grid
                param_grid = None  # Use default in the classifier class

                # Run hyperparameter optimization
                self.progress_signal.emit(40)
                optimization_result = classifier.optimize_hyperparameters(self.X, self.y, param_grid)

                # Update progress
                self.progress_signal.emit(80)

                # Train final model with best params
                results = classifier.train(self.X, self.y,
                                          test_size=self.model_config.get('test_size', 0.2))

                # Combine optimization and training results
                results.update(optimization_result)
            else:
                # Just train with current parameters
                self.progress_signal.emit(50)
                results = classifier.train(self.X, self.y,
                                          test_size=self.model_config.get('test_size', 0.2))

            # Add task type to results
            results["is_regression"] = self.is_regression

            # Update progress
            self.progress_signal.emit(100)

            print(f"Training completed successfully")
            print(f"Results: {results.keys()}")
            print(f"Metrics: {results.get('metrics', {})}")
            print("===================================\n")

            # Emit results
            self.result_signal.emit(results)

        except Exception as e:
            import traceback
            print(f"Error in training thread: {e}")
            print(traceback.format_exc())
            self.result_signal.emit({"error": str(e)})


class ModelTrainingTab(QWidget):
    model_ready = pyqtSignal(bool)

    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data

        # Debug print to see what's in app_data at initialization
        print("\n==== MODEL TAB INIT ====")
        print(f"app_data keys: {list(self.app_data.keys())}")
        if 'feature_matrix' in self.app_data and self.app_data['feature_matrix'] is not None:
            print(f"Feature matrix exists with shape: {self.app_data['feature_matrix'].shape}")
            print(f"Feature matrix columns: {self.app_data['feature_matrix'].columns.tolist()}")

            # Store metadata columns for later use
            metadata_cols = [col for col in self.app_data['feature_matrix'].columns
                            if col in ['concentration', 'antibiotic']]
            if metadata_cols:
                print(f"Found metadata columns: {metadata_cols}")
                self.app_data["metadata_columns"] = metadata_cols
        else:
            print("No feature_matrix in app_data at initialization")
        print("===========================\n")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)

        # Data & Model Configuration Section
        config_group = QGroupBox("Model Configuration")
        config_layout = QGridLayout()

        # Target variable selection
        config_layout.addWidget(QLabel("Target Variable:"), 0, 0)
        self.target_combo = QComboBox()
        config_layout.addWidget(self.target_combo, 0, 1)

        # DIRECT FIX: Immediately populate target dropdown with metadata columns if available
        if self.app_data.get("metadata_columns") is not None:
            metadata_columns = self.app_data["metadata_columns"]
            if metadata_columns:
                self.target_combo.addItems(metadata_columns)
                print(f"Added metadata columns to target dropdown during init_ui: {metadata_columns}")
        elif self.app_data.get("feature_matrix") is not None:
            feature_matrix = self.app_data["feature_matrix"]
            metadata_columns = [col for col in feature_matrix.columns if col in ['concentration', 'antibiotic']]
            if metadata_columns:
                self.target_combo.addItems(metadata_columns)
                self.app_data["metadata_columns"] = metadata_columns
                print(f"Added metadata columns from feature_matrix during init_ui: {metadata_columns}")

        # Model type selection
        config_layout.addWidget(QLabel("Model Type:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest (rf)",
            "Gradient Boosting (gb)",
            "Support Vector Machine (svm)",
            "XGBoost (xgb)",
            "LightGBM (lgb)"
        ])
        self.model_combo.currentIndexChanged.connect(self.update_model_options)
        config_layout.addWidget(self.model_combo, 1, 1)

        # Train/Test split
        config_layout.addWidget(QLabel("Test Split:"), 2, 0)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 50)
        self.test_spin.setValue(20)
        self.test_spin.setSuffix("%")
        config_layout.addWidget(self.test_spin, 2, 1)

        # Removed CV Folds section

        # Hyperparameter optimization
        self.optimize_check = QCheckBox("Hyperparameter Optimization")
        config_layout.addWidget(self.optimize_check, 3, 0, 1, 2)

        # Model specific params section (will be populated based on model type)
        self.params_frame = QFrame()
        self.params_layout = QGridLayout(self.params_frame)
        config_layout.addWidget(self.params_frame, 4, 0, 1, 2)

        # Training button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        config_layout.addWidget(self.train_btn, 5, 0, 1, 2)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        config_layout.addWidget(self.progress_bar, 6, 0, 1, 2)

        config_group.setLayout(config_layout)
        splitter.addWidget(config_group)

        # Training Results Section with Tabs
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()

        # Tabs for different result views
        self.result_tabs = QTabWidget()

        # Metrics tab
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # Metrics table
        self.metrics_table = QTableWidget(5, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setItem(0, 0, QTableWidgetItem("Accuracy"))
        self.metrics_table.setItem(1, 0, QTableWidgetItem("Precision"))
        self.metrics_table.setItem(2, 0, QTableWidgetItem("Recall"))
        self.metrics_table.setItem(3, 0, QTableWidgetItem("F1 Score"))
        self.metrics_table.setItem(4, 0, QTableWidgetItem("ROC AUC"))
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)

        self.result_tabs.addTab(self.metrics_tab, "Performance Metrics")

        # Confusion Matrix & ROC tab
        self.viz_tab = QWidget()
        viz_layout = QHBoxLayout(self.viz_tab)

        # Confusion matrix plot
        self.cm_figure = Figure(figsize=(4, 4), dpi=100)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        viz_layout.addWidget(self.cm_canvas)

        # ROC curve plot
        self.roc_figure = Figure(figsize=(4, 4), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        viz_layout.addWidget(self.roc_canvas)

        self.result_tabs.addTab(self.viz_tab, "Evaluation Plots")

        # Feature Importance tab
        self.feature_tab = QWidget()
        feature_layout = QVBoxLayout(self.feature_tab)

        self.feature_figure = Figure(figsize=(6, 4), dpi=100)
        self.feature_canvas = FigureCanvas(self.feature_figure)
        feature_layout.addWidget(self.feature_canvas)

        self.result_tabs.addTab(self.feature_tab, "Feature Importance")

        # Hyperparameter optimization results tab
        self.hyperparam_tab = QWidget()
        hyperparam_layout = QVBoxLayout(self.hyperparam_tab)

        self.hyperparam_table = QTableWidget()
        self.hyperparam_table.setColumnCount(2)
        self.hyperparam_table.setHorizontalHeaderLabels(["Parameter", "Optimal Value"])
        self.hyperparam_table.horizontalHeader().setStretchLastSection(True)
        hyperparam_layout.addWidget(self.hyperparam_table)

        self.result_tabs.addTab(self.hyperparam_tab, "Hyperparameters")

        results_layout.addWidget(self.result_tabs)

        # Save model and continue buttons
        buttons_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        self.continue_btn = QPushButton("Continue to Analysis")
        self.continue_btn.clicked.connect(self.continue_to_analysis)
        self.continue_btn.setEnabled(False)
        buttons_layout.addWidget(self.continue_btn)

        results_layout.addLayout(buttons_layout)

        results_group.setLayout(results_layout)
        splitter.addWidget(results_group)

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Initialize model-specific parameters
        self.update_model_options()

    def update_model_options(self):
        """Update parameter options based on selected model"""
        # Clear references to parameter widgets
        self.n_estimators_spin = None
        self.max_depth_spin = None
        self.learning_rate_spin = None
        self.kernel_combo = None
        self.c_spin = None
        self.gamma_combo = None

        # Clear current params layout
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        model_type = self.model_combo.currentText()

        if "Random Forest" in model_type:
            # Add RF specific params
            self.params_layout.addWidget(QLabel("Number of Trees:"), 0, 0)
            self.n_estimators_spin = QSpinBox()
            self.n_estimators_spin.setRange(10, 1000)
            self.n_estimators_spin.setValue(100)
            self.params_layout.addWidget(self.n_estimators_spin, 0, 1)

            self.params_layout.addWidget(QLabel("Max Depth:"), 1, 0)
            self.max_depth_spin = QSpinBox()
            self.max_depth_spin.setRange(1, 100)
            self.max_depth_spin.setValue(10)
            self.params_layout.addWidget(self.max_depth_spin, 1, 1)

        elif "Gradient Boosting" in model_type or "XGBoost" in model_type or "LightGBM" in model_type:
            # Add GB specific params
            self.params_layout.addWidget(QLabel("Number of Trees:"), 0, 0)
            self.n_estimators_spin = QSpinBox()
            self.n_estimators_spin.setRange(10, 1000)
            self.n_estimators_spin.setValue(100)
            self.params_layout.addWidget(self.n_estimators_spin, 0, 1)

            self.params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
            self.learning_rate_spin = QDoubleSpinBox()
            self.learning_rate_spin.setRange(0.001, 1.0)
            self.learning_rate_spin.setValue(0.1)
            self.learning_rate_spin.setSingleStep(0.01)
            self.params_layout.addWidget(self.learning_rate_spin, 1, 1)

            self.params_layout.addWidget(QLabel("Max Depth:"), 2, 0)
            self.max_depth_spin = QSpinBox()
            self.max_depth_spin.setRange(1, 100)
            self.max_depth_spin.setValue(3)
            self.params_layout.addWidget(self.max_depth_spin, 2, 1)

        elif "Support Vector Machine" in model_type:
            # Add SVM specific params
            self.params_layout.addWidget(QLabel("Kernel:"), 0, 0)
            self.kernel_combo = QComboBox()
            self.kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
            self.params_layout.addWidget(self.kernel_combo, 0, 1)

            self.params_layout.addWidget(QLabel("C:"), 1, 0)
            self.c_spin = QDoubleSpinBox()
            self.c_spin.setRange(0.1, 100.0)
            self.c_spin.setValue(1.0)
            self.c_spin.setSingleStep(0.1)
            self.params_layout.addWidget(self.c_spin, 1, 1)

            self.params_layout.addWidget(QLabel("Gamma:"), 2, 0)
            self.gamma_combo = QComboBox()
            self.gamma_combo.addItems(["scale", "auto", "0.1", "0.01", "0.001"])
            self.params_layout.addWidget(self.gamma_combo, 2, 1)

    def update_column_list(self):
        """Update list of available columns from dataset"""
        try:
            # DIRECT APPROACH: First check if metadata_columns is directly available
            if self.app_data.get("metadata_columns") is not None:
                metadata_columns = self.app_data["metadata_columns"]
                self.target_combo.clear()

                if metadata_columns:
                    self.target_combo.addItems(metadata_columns)
                    print(f"Using metadata_columns from app_data: {metadata_columns}")
                    return

            # Check if we have a feature matrix (new format) with metadata columns
            if self.app_data.get("feature_matrix") is not None:
                feature_matrix = self.app_data["feature_matrix"]
                self.target_combo.clear()

                # Add metadata columns like concentration and antibiotic
                metadata_columns = [col for col in feature_matrix.columns if col in ['concentration', 'antibiotic', 'path']]
                if metadata_columns:
                    # Only add concentration and antibiotic as target variables (not path)
                    target_columns = [col for col in metadata_columns if col in ['concentration', 'antibiotic']]
                    if target_columns:
                        self.target_combo.addItems(target_columns)
                        print(f"Found target columns in feature_matrix: {target_columns}")
                        # Store for future use
                        self.app_data["metadata_columns"] = metadata_columns
                        return
                    else:
                        print("No target columns found in metadata columns")
                else:
                    print("No metadata columns found in feature_matrix")

            # Fallback to processed_data if no feature_matrix is available
            elif self.app_data.get("processed_data") is not None:
                data = self.app_data["processed_data"]
                self.target_combo.clear()
                self.target_combo.addItems(data.columns)
                print(f"Using columns from processed_data: {data.columns.tolist()}")
            else:
                print("No data sources available for target variables")
                # Add a default item so the dropdown isn't empty
                self.target_combo.clear()
                self.target_combo.addItem("No target variables available")
        except Exception as e:
            print(f"Error in update_column_list: {str(e)}")
            # Add a default item so the dropdown isn't empty
            self.target_combo.clear()
            self.target_combo.addItem("Error loading target variables")

    def train_model(self):
        """Train the model with selected configuration"""
        # Get selected target
        target_col = self.target_combo.currentText()
        if not target_col:
            QMessageBox.warning(self, "No Target", "Please select a target variable for training.")
            return

        # Check if we have a feature matrix (new format)
        if self.app_data.get("feature_matrix") is not None:
            feature_matrix = self.app_data["feature_matrix"]

            # Check if target column exists in the feature matrix
            if target_col not in feature_matrix.columns:
                QMessageBox.warning(self, "Missing Target",
                                  f"Target column '{target_col}' not found in the feature matrix.")
                return

            # Get all feature columns (exclude metadata columns)
            feature_cols = [col for col in feature_matrix.columns
                           if col.startswith(('FFT', 'PC', 'Window'))]

            # Check if we have any features
            if len(feature_cols) == 0:
                QMessageBox.warning(self, "No Features",
                                  "No feature columns found in the dataset. Please extract features first.")
                return

            # Debug information
            print(f"Using {len(feature_cols)} features for training")
            print(f"First few feature names: {feature_cols[:5]}")

            # Get feature matrix and target vector
            X = feature_matrix[feature_cols].values

            # Convert target to appropriate type based on task
            if target_col == 'concentration':
                # For regression, ensure target is numeric
                try:
                    # First try to convert to float directly
                    y = pd.to_numeric(feature_matrix[target_col], errors='coerce')

                    # Check for NaN values after conversion
                    if y.isna().any():
                        print(f"Warning: {y.isna().sum()} NaN values found in concentration after conversion. Dropping these rows.")
                        # Get indices of non-NaN values
                        valid_indices = ~y.isna()
                        # Filter X and y
                        X = X[valid_indices]
                        y = y[valid_indices].values
                    else:
                        y = y.values

                    print(f"Converted concentration to float: {y[:5]}")
                    print(f"Concentration range: {np.min(y)} to {np.max(y)}, mean: {np.mean(y)}, std: {np.std(y)}")

                    # Check if all values are the same
                    if np.min(y) == np.max(y):
                        QMessageBox.warning(self, "Data Variation Error",
                                          "All concentration values are identical. Regression models require variation in the target variable.")
                        return
                except Exception as e:
                    print(f"Error converting concentration to float: {e}")
                    QMessageBox.warning(self, "Data Type Error",
                                      f"Could not convert concentration to numeric values: {e}")
                    return
            else:
                # For classification, keep as is (could be string or numeric)
                y = feature_matrix[target_col].values
                print(f"Using classification target: {y[:5]}")
                print(f"Unique classes: {np.unique(y)}")

                # Check if we have enough samples for each class
                unique_classes, class_counts = np.unique(y, return_counts=True)
                min_samples = np.min(class_counts)
                if min_samples < 3:
                    QMessageBox.warning(self, "Warning",
                                     f"Some classes have very few samples (minimum: {min_samples}). This may affect model performance.")

            # Store selected features for later use
            self.app_data["selected_features"] = feature_cols

        # Fallback to processed_data if no feature_matrix is available
        elif self.app_data.get("processed_data") is not None and self.app_data.get("selected_features") is not None:
            # Prepare data
            data = self.app_data["processed_data"]
            selected_features = self.app_data["selected_features"]

            # Check if features and target are in the dataset
            if target_col not in data.columns:
                QMessageBox.warning(self, "Missing Target",
                                  f"Target column '{target_col}' not found in the dataset.")
                return

            missing_features = [f for f in selected_features if f not in data.columns]
            if missing_features:
                QMessageBox.warning(self, "Missing Features",
                                  f"Some features are missing in the dataset: {missing_features}")
                return

            # Get feature matrix and target vector
            X = data[selected_features].values
            y = data[target_col].values
        else:
            QMessageBox.warning(self, "Missing Data",
                              "No feature matrix or processed data available. Please extract features first.")
            return

        # Store target variable name
        self.app_data["target_variable"] = target_col

        # Determine if this is a regression or classification task based on target variable
        is_regression = False
        if target_col == 'concentration':
            is_regression = True
            print(f"Detected regression task for target '{target_col}'")

            # Check if target values are numeric
            if not pd.api.types.is_numeric_dtype(y):
                try:
                    # Try to convert to numeric
                    y = pd.to_numeric(y, errors='coerce')
                    # Check for NaN values after conversion
                    if y.isna().any():
                        print(f"Warning: {y.isna().sum()} NaN values found after converting to numeric. Dropping these rows.")
                        # Get indices of non-NaN values
                        valid_indices = ~y.isna()
                        # Filter X and y
                        X = X[valid_indices]
                        y = y[valid_indices]
                except Exception as e:
                    print(f"Error converting target to numeric: {e}")
                    QMessageBox.critical(self, "Error",
                                      f"The target column '{target_col}' contains non-numeric values that cannot be used for regression.")
                    return

            # Print target variable statistics
            print(f"Target variable statistics: min={y.min()}, max={y.max()}, mean={y.mean()}, std={y.std()}")
            print(f"Number of samples after preprocessing: {len(y)}")

            # Check if we have enough variation in the target variable
            # Use np.unique for numpy arrays
            unique_values = np.unique(y)
            if len(unique_values) < 3:
                QMessageBox.warning(self, "Warning",
                                 f"The target column '{target_col}' has only {len(unique_values)} unique values. Regression models work best with more variation.")

            # Inform the user about the task type
            QMessageBox.information(self, "Regression Task",
                                  f"Training a regression model for '{target_col}' prediction.")
        else:
            print(f"Detected classification task for target '{target_col}'")
            # Inform the user about the task type
            QMessageBox.information(self, "Classification Task",
                                  f"Training a classification model for '{target_col}' detection.")

        # Configure model
        model_type = self.model_combo.currentText()
        # Extract abbreviation from model type text
        model_type_code = model_type.split("(")[1].strip(")")

        model_config = {
            "model_type": model_type_code,
            "test_size": self.test_spin.value() / 100,  # Convert from percentage
            "optimize": self.optimize_check.isChecked()
        }

        # Add model-specific parameters - check if attributes exist AND are valid widgets
        try:
            if hasattr(self, "n_estimators_spin") and self.n_estimators_spin is not None:
                model_config["n_estimators"] = self.n_estimators_spin.value()
        except RuntimeError as e:
            print(f"Error accessing n_estimators_spin: {e}")

        try:
            if hasattr(self, "max_depth_spin") and self.max_depth_spin is not None:
                model_config["max_depth"] = self.max_depth_spin.value()
        except RuntimeError as e:
            print(f"Error accessing max_depth_spin: {e}")

        try:
            if hasattr(self, "learning_rate_spin") and self.learning_rate_spin is not None:
                model_config["learning_rate"] = self.learning_rate_spin.value()
        except RuntimeError as e:
            print(f"Error accessing learning_rate_spin: {e}")

        try:
            if hasattr(self, "kernel_combo") and self.kernel_combo is not None:
                model_config["kernel"] = self.kernel_combo.currentText()
        except RuntimeError as e:
            print(f"Error accessing kernel_combo: {e}")

        try:
            if hasattr(self, "c_spin") and self.c_spin is not None:
                model_config["C"] = self.c_spin.value()
        except RuntimeError as e:
            print(f"Error accessing c_spin: {e}")

        try:
            if hasattr(self, "gamma_combo") and self.gamma_combo is not None:
                gamma = self.gamma_combo.currentText()
                # Convert numeric strings to float
                if gamma not in ["scale", "auto"]:
                    try:
                        gamma = float(gamma)
                    except ValueError:
                        pass
                model_config["gamma"] = gamma
        except RuntimeError as e:
            print(f"Error accessing gamma_combo: {e}")

        # Reset progress bar
        self.progress_bar.setValue(0)

        # Store model config
        self.app_data["model_config"] = model_config
        self.app_data["is_regression"] = is_regression

        # Create and start training thread
        self.training_thread = TrainingThread(model_config, X, y, is_regression)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.result_signal.connect(self.process_results)
        self.training_thread.start()

        # Disable train button during training
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")

    @pyqtSlot(int)
    def update_progress(self, value):
        """Update the progress bar during training"""
        self.progress_bar.setValue(value)

    @pyqtSlot(dict)
    def process_results(self, results):
        """Process and display training results"""
        # Re-enable train button
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Train Model")

        # Check for errors
        if "error" in results:
            print(f"Training error: {results['error']}")
            QMessageBox.critical(self, "Training Error", f"Error during model training: {results['error']}")
            return

        # Store results
        self.app_data["model_results"] = results

        # Check if this is a regression or classification task
        is_regression = results.get("is_regression", False)

        # Update metrics table based on task type
        metrics = results.get("metrics", {})
        if metrics:
            # Clear existing metrics
            self.metrics_table.clear()

            if is_regression:
                # Set up regression metrics table
                self.metrics_table.setRowCount(3)
                self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
                self.metrics_table.setItem(0, 0, QTableWidgetItem("R² Score"))
                self.metrics_table.setItem(1, 0, QTableWidgetItem("Mean Squared Error"))
                self.metrics_table.setItem(2, 0, QTableWidgetItem("Root MSE"))

                # Expand regression metrics table to show more metrics
                self.metrics_table.setRowCount(5)
                self.metrics_table.setItem(0, 0, QTableWidgetItem("R² Score"))
                self.metrics_table.setItem(1, 0, QTableWidgetItem("Mean Squared Error"))
                self.metrics_table.setItem(2, 0, QTableWidgetItem("Root MSE"))
                self.metrics_table.setItem(3, 0, QTableWidgetItem("Explained Variance"))
                self.metrics_table.setItem(4, 0, QTableWidgetItem("Mean Absolute Error"))

                # Fill in regression metrics
                self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics.get('r2', 0):.4f}"))
                self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics.get('mse', 0):.4f}"))
                self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{metrics.get('rmse', 0):.4f}"))
                self.metrics_table.setItem(3, 1, QTableWidgetItem(f"{metrics.get('explained_variance', 0):.4f}"))
                self.metrics_table.setItem(4, 1, QTableWidgetItem(f"{metrics.get('mean_absolute_error', 0):.4f}"))
            else:
                # Set up classification metrics table
                # Determine how many metrics to show based on what's available
                available_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
                metrics_to_show = [m for m in available_metrics if m in metrics]

                self.metrics_table.setRowCount(len(metrics_to_show))
                self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])

                # Fill in classification metrics
                for i, metric_name in enumerate(metrics_to_show):
                    # Format metric name for display
                    display_name = metric_name.replace('_', ' ').title()
                    if metric_name == "roc_auc":
                        display_name = "ROC AUC"
                    elif metric_name == "f1":
                        display_name = "F1 Score"

                    self.metrics_table.setItem(i, 0, QTableWidgetItem(display_name))
                    self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{metrics.get(metric_name, 0):.4f}"))

        # Update visualization based on task type
        if is_regression:
            # For regression, show scatter plot of actual vs predicted values
            self.update_regression_plot(results)

            # Hide classification-specific tabs
            self.result_tabs.setTabVisible(1, False)  # Hide Evaluation Plots tab
        else:
            # For classification, show confusion matrix
            self.update_confusion_matrix(results)

            # Only show ROC curve for binary classification
            y_test = results.get("y_test", [])
            if len(y_test) > 0 and len(np.unique(y_test)) == 2:
                # Binary classification - can show ROC curve
                self.update_roc_curve(results)
                # Show classification-specific tabs
                self.result_tabs.setTabVisible(1, True)  # Show Evaluation Plots tab
            else:
                # Multiclass classification - hide ROC curve tab
                self.result_tabs.setTabVisible(1, False)  # Hide Evaluation Plots tab

        # Update feature importance plot (works for both regression and classification)
        self.update_feature_importance(results)

        # Update hyperparameter results
        best_params = results.get("best_params", {})
        if best_params:
            self.hyperparam_table.setRowCount(len(best_params))
            for i, (param, value) in enumerate(best_params.items()):
                self.hyperparam_table.setItem(i, 0, QTableWidgetItem(param))
                self.hyperparam_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Enable save and continue buttons
        self.save_btn.setEnabled(True)
        self.continue_btn.setEnabled(True)

        # Signal that model is ready
        self.model_ready.emit(True)

    def update_confusion_matrix(self, results):
        """Update confusion matrix visualization"""
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        # Get confusion matrix from results
        metrics = results.get("metrics", {})
        cm = metrics.get("confusion_matrix", [])

        if cm and len(cm) > 0:
            # Convert to numpy array if it's a list
            if isinstance(cm, list):
                cm = np.array(cm)

            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            self.cm_figure.colorbar(im, ax=ax)

            # Set labels based on matrix size
            if cm.shape[0] == 2:  # Binary classification
                classes = ["Negative", "Positive"]
            else:  # Multiclass classification
                classes = [str(i) for i in range(cm.shape[0])]

            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            ax.set_title("Confusion Matrix")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")

        self.cm_canvas.draw()

    def update_roc_curve(self, results):
        """Update ROC curve visualization for binary classification"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        y_test = results.get("y_test", [])
        y_pred_proba = results.get("y_pred_proba", [])

        # Check if we have binary classification data
        if len(y_test) > 0 and len(y_pred_proba) > 0 and len(np.unique(y_test)) == 2:
            try:
                # For binary classification, we need the probability of the positive class
                # If y_pred_proba is 2D (one column per class), take the second column
                if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                    y_pred_proba_binary = y_pred_proba[:, 1]
                else:
                    # Otherwise use as is (assuming it's already the probability of the positive class)
                    y_pred_proba_binary = y_pred_proba

                # Plot ROC curve
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba_binary)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
            except Exception as e:
                print(f"Error plotting ROC curve: {e}")
                ax.text(0.5, 0.5, f"Error plotting ROC curve: {e}", ha="center", va="center", wrap=True)
        else:
            ax.text(0.5, 0.5, "ROC curve only available for binary classification", ha="center", va="center")

        self.roc_canvas.draw()

    def update_regression_plot(self, results):
        """Update regression plot with actual vs predicted values"""
        self.cm_figure.clear()  # Use the confusion matrix figure for regression plot
        ax = self.cm_figure.add_subplot(111)

        y_test = results.get("y_test", [])
        y_pred = results.get("y_pred", [])

        if len(y_test) > 0 and len(y_pred) > 0:
            # Create scatter plot of actual vs predicted values
            ax.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')

            # Add perfect prediction line (y=x)
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

            # Add labels and title
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            ax.legend()

            # Add metrics as text
            r2 = results.get("metrics", {}).get("r2", 0)
            mse = results.get("metrics", {}).get("mse", 0)
            explained_var = results.get("metrics", {}).get("explained_variance", 0)

            # Create a text box with multiple metrics
            metrics_text = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nExpl. Var = {explained_var:.4f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No regression data available", ha="center", va="center")

        self.cm_canvas.draw()

        # Also clear the ROC curve plot since we're not using it for regression
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        # Create a residual plot (actual - predicted)
        if len(y_test) > 0 and len(y_pred) > 0:
            residuals = y_test - y_pred
            ax.scatter(y_pred, residuals, alpha=0.7, color='green')
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No regression data available", ha="center", va="center")

        self.roc_canvas.draw()

    def update_feature_importance(self, results):
        """Update feature importance visualization"""
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        feature_importances = results.get("feature_importances")
        selected_features = self.app_data.get("selected_features", [])

        if feature_importances is not None and len(selected_features) > 0:
            # Sort features by importance
            indices = np.argsort(feature_importances)

            # Plot top 20 features or all if less than 20
            n_features = min(20, len(selected_features))
            plt_indices = indices[-n_features:]

            # Get feature names for plotting
            feature_names = [selected_features[i] for i in plt_indices]
            feature_values = [feature_importances[i] for i in plt_indices]

            # Horizontal bar chart
            ax.barh(range(n_features), feature_values, align='center')
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(feature_names)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
        else:
            ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")

        self.feature_canvas.draw()

    def save_model(self):
        """Save the trained model"""
        # This would open a dialog to save the model to a file
        # In this example, we'll just print a message
        print("Model saving functionality would be implemented here")

    def continue_to_analysis(self):
        """Continue to the results analysis tab"""
        # Signal to main window to switch to results tab
        self.model_ready.emit(True)

    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)

        try:
            # Debug print to see what's in app_data
            print("\n==== MODEL TAB SHOW EVENT ====")
            print(f"app_data keys: {list(self.app_data.keys())}")
            if 'feature_matrix' in self.app_data and self.app_data['feature_matrix'] is not None:
                print(f"Feature matrix exists with shape: {self.app_data['feature_matrix'].shape}")
                print(f"Feature matrix columns: {self.app_data['feature_matrix'].columns.tolist()}")
            else:
                print("No feature_matrix in app_data")
            print("==============================\n")

            # DIRECT APPROACH: Add metadata columns to target dropdown
            # This will run regardless of update_column_list
            if self.app_data.get("feature_matrix") is not None:
                feature_matrix = self.app_data["feature_matrix"]

                # Clear existing items
                self.target_combo.clear()

                # Add metadata columns - use a more flexible approach
                important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
                metadata_columns = []

                # First check for exact matches
                for col in feature_matrix.columns:
                    if col in important_metadata:
                        metadata_columns.append(col)

                # Then check for case-insensitive matches if we didn't find any
                if not metadata_columns:
                    for col in feature_matrix.columns:
                        if col.lower() in [meta.lower() for meta in important_metadata]:
                            metadata_columns.append(col)

                if metadata_columns:
                    self.target_combo.addItems(metadata_columns)
                    print(f"Added metadata columns to target dropdown: {metadata_columns}")
                    # Store metadata columns for future use
                    self.app_data["metadata_columns"] = metadata_columns
                    # Show a message box to inform the user
                    QMessageBox.information(self, "Target Variables Available",
                                         f"Found metadata columns that can be used as target variables: {', '.join(metadata_columns)}")
                else:
                    print("No metadata columns found in feature matrix")
                    # If no metadata columns found, check all columns and suggest potential targets
                    potential_targets = []
                    for col in feature_matrix.columns:
                        # Skip columns that look like features
                        if not any(col.startswith(prefix) for prefix in ['basic_', 'peak_', 'shape_', 'derivative_', 'area_']):
                            potential_targets.append(col)

                    if potential_targets:
                        print(f"Potential target columns: {potential_targets}")
                        self.target_combo.addItems(potential_targets)
                        QMessageBox.information(self, "Potential Target Variables",
                                             f"No standard metadata columns found, but these columns might be usable as targets: {', '.join(potential_targets)}")
            else:
                # Fallback to update_column_list if no feature_matrix
                self.update_column_list()
        except Exception as e:
            print(f"Error in showEvent: {str(e)}")
            # Don't crash, just continue