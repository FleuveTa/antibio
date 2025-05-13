from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QFileDialog, QGroupBox, 
                             QGridLayout, QMessageBox, QTableWidget, QTableWidgetItem,
                             QSplitter, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from pathlib import Path

from app.core.experiment_manager import ExperimentManager
from app.core.preprocessing import apply_transformers
from app.core.feature_eng import apply_pca_to_wide_data, apply_fft_to_wide_data, apply_windowed_integral_to_wide_data


class PredictionTab(QWidget):
    """Tab for making predictions on new data using a trained model"""
    
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        
        # Initialize experiment manager
        try:
            self.experiment_manager = ExperimentManager(base_dir="data")
            print("Prediction tab: Experiment manager initialized")
        except Exception as e:
            print(f"Error initializing experiment manager in prediction tab: {e}")
            QMessageBox.critical(self, "Error", f"Error initializing experiment manager: {e}")
            self.experiment_manager = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        
        # Header with instructions
        header_label = QLabel("Predict New Samples Using Trained Models")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        instruction_label = QLabel(
            "This tab allows you to use a trained model to make predictions on new data samples. "
            "First load an experiment with a trained model, then import new data for prediction.\n\n"
            "If you haven't trained a model yet, you can create a new experiment in the Data Import tab, "
            "then follow the workflow through preprocessing, feature extraction, and model training."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(instruction_label)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Model selection and data import
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Experiment selection
        exp_group = QGroupBox("Load Experiment with Trained Model")
        exp_layout = QGridLayout()
        
        # Add description
        description_label = QLabel(
            "Select an existing experiment to use its trained model for prediction. "
            "You can also delete experiments that are no longer needed."
        )
        description_label.setWordWrap(True)
        exp_layout.addWidget(description_label, 0, 0, 1, 3)
        
        # Add button to create new experiment
        new_exp_btn = QPushButton("➕ Create New Experiment")
        new_exp_btn.setToolTip("Switch to Data Import tab to create a new experiment")
        new_exp_btn.clicked.connect(self.go_to_data_import_tab)
        new_exp_btn.setStyleSheet("background-color: #e6f2ff;")
        exp_layout.addWidget(new_exp_btn, 1, 0, 1, 3)
        
        exp_layout.addWidget(QLabel("Experiment:"), 2, 0)
        self.exp_combo = QComboBox()
        self.exp_combo.currentIndexChanged.connect(self.experiment_selected)
        exp_layout.addWidget(self.exp_combo, 2, 1)
        
        # Control buttons in a horizontal layout
        controls_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_experiments)
        controls_layout.addWidget(self.refresh_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setToolTip("Delete selected experiment")
        self.delete_btn.clicked.connect(self.delete_experiment)
        self.delete_btn.setStyleSheet("background-color: #ffcccc;")
        self.delete_btn.setEnabled(False)
        controls_layout.addWidget(self.delete_btn)
        
        exp_layout.addLayout(controls_layout, 2, 2)
        
        # Model info
        exp_layout.addWidget(QLabel("Model Type:"), 3, 0)
        self.model_type_label = QLabel("None")
        self.model_type_label.setStyleSheet("font-weight: bold;")
        exp_layout.addWidget(self.model_type_label, 3, 1)
        
        exp_layout.addWidget(QLabel("Target Variable:"), 4, 0)
        self.target_label = QLabel("None")
        self.target_label.setStyleSheet("font-weight: bold;")
        exp_layout.addWidget(self.target_label, 4, 1)
        
        exp_layout.addWidget(QLabel("Performance:"), 5, 0)
        self.accuracy_label = QLabel("None")
        self.accuracy_label.setStyleSheet("font-weight: bold;")
        exp_layout.addWidget(self.accuracy_label, 5, 1)
        
        exp_group.setLayout(exp_layout)
        left_layout.addWidget(exp_group)
        
        # Data import
        data_group = QGroupBox("Import New Data for Prediction")
        data_layout = QGridLayout()
        
        data_layout.addWidget(QLabel("Data File:"), 0, 0)
        self.file_path = QLabel("No file selected")
        data_layout.addWidget(self.file_path, 0, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_files)
        data_layout.addWidget(self.browse_btn, 0, 2)
        
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("background-color: #d4f7d4; font-weight: bold; padding: 5px;")
        data_layout.addWidget(self.predict_btn, 1, 0, 1, 3)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        # Status section
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Please select an experiment with a trained model")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results table
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)  # Sample ID, Actual (if available), Predicted
        self.results_table.setHorizontalHeaderLabels(["Sample ID", "Actual (if available)", "Predicted"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        right_layout.addWidget(viz_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])  # Initial sizes
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
        # Load experiments
        self.load_experiments()
    
    def load_experiments(self):
        """Load list of experiments with trained models"""
        try:
            self.exp_combo.clear()
            self.exp_combo.addItem("Select an experiment...", None)
            
            # Get all experiments
            experiments = self.experiment_manager.list_experiments()
            
            # Filter experiments that have trained models
            models_dir = Path("data/models")
            experiments_with_models = []
            
            for exp in experiments:
                exp_id = exp['id']
                exp_model_dir = models_dir / f"experiment_{exp_id}"
                
                if exp_model_dir.exists() and any(exp_model_dir.iterdir()):
                    experiments_with_models.append(exp)
                    self.exp_combo.addItem(f"{exp['name']} (ID: {exp_id})", exp_id)
            
            if not experiments_with_models:
                self.status_label.setText("No experiments with trained models found. Train a model first.")
            else:
                self.status_label.setText(f"Found {len(experiments_with_models)} experiments with trained models.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading experiments: {str(e)}")
            print(f"Error loading experiments: {str(e)}")
    
    def experiment_selected(self, index):
        """Handle experiment selection"""
        if index <= 0:  # First item is placeholder
            self.clear_model_info()
            self.predict_btn.setEnabled(False)
            return
        
        experiment_id = self.exp_combo.currentData()
        if not experiment_id:
            return
        
        try:
            # Load the experiment
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                QMessageBox.warning(self, "Warning", "Could not find experiment details.")
                return
            
            # First check if we have any models in the database
            from pathlib import Path
            import sqlite3
            
            # Check models directory for available model types
            models_dir = Path("data/models") / f"experiment_{experiment_id}"
            if not models_dir.exists():
                QMessageBox.warning(self, "Warning", "No trained models found for this experiment.")
                self.clear_model_info()
                return
            
            # Look for model files directly
            model_files = list(models_dir.glob("*_model.pkl"))
            
            if not model_files:
                QMessageBox.warning(self, "Warning", "No model files found for this experiment.")
                self.clear_model_info()
                return
            
            # Extract model types from filenames
            available_models = []
            for model_file in model_files:
                # Extract model type from filename (remove _model.pkl)
                model_type = model_file.name.replace('_model.pkl', '')
                available_models.append(model_type)
            
            print(f"Available models for experiment {experiment_id}: {available_models}")
            
            # Load the first available model
            if available_models:
                model_type = available_models[0]
                
                try:
                    model, metrics = self.experiment_manager.load_model(experiment_id, model_type)
                    
                    if not model:
                        QMessageBox.warning(self, "Warning", f"Failed to load {model_type} model for this experiment.")
                        self.clear_model_info()
                        return
                    
                    # If we don't have metrics from load_model, try to load from metrics file
                    if not metrics:
                        metrics_file = models_dir / f"{model_type}_metrics.json"
                        if metrics_file.exists():
                            import json
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                print(f"Loaded metrics from file: {metrics}")
                            except Exception as e:
                                print(f"Error reading metrics file: {e}")
                                metrics = {}
                    
                    # Update UI with model info
                    self.model_type_label.setText(model_type)
                    
                    target_col = metrics.get('target_column', 'Unknown')
                    self.target_label.setText(target_col)
                    
                    # Hiển thị thông tin metrics thay vì chỉ hiển thị "Model loaded successfully"
                    is_regression = metrics.get('is_regression', target_col == 'concentration')
                    
                    if is_regression:
                        # Hiển thị R² cho mô hình regression
                        r2_value = metrics.get('r2', 0)
                        try:
                            r2_display = f"R² = {float(r2_value):.4f}"
                        except (ValueError, TypeError):
                            r2_display = "R² not available"
                        self.accuracy_label.setText(r2_display)
                    else:
                        # Hiển thị accuracy cho mô hình phân loại
                        accuracy_value = metrics.get('accuracy', 0)
                        try:
                            accuracy_display = f"Accuracy = {float(accuracy_value):.4f}"
                        except (ValueError, TypeError):
                            accuracy_display = "Accuracy not available"
                        self.accuracy_label.setText(accuracy_display)
                    
                    # Store model and metadata
                    self.app_data["current_experiment_id"] = experiment_id
                    self.app_data["trained_model"] = model
                    self.app_data["model_metrics"] = metrics
                    self.app_data["model_type"] = model_type
                    
                    # Enable prediction if we have a file
                    self.predict_btn.setEnabled(self.file_path.text() != "No file selected")
                    
                    # Update status
                    self.status_label.setText(f"Loaded model: {model_type} for experiment: {experiment['name']}")
                    
                    # Enable delete button
                    self.delete_btn.setEnabled(True)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error loading {model_type} model: {str(e)}")
                    print(f"Error loading {model_type} model: {str(e)}")
                    self.clear_model_info()
            else:
                QMessageBox.warning(self, "Warning", "No trained models found for this experiment.")
                self.clear_model_info()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            self.clear_model_info()
    
    def clear_model_info(self):
        """Clear model information"""
        self.model_type_label.setText("None")
        self.target_label.setText("None")
        self.accuracy_label.setText("None")
        
        # Clear app data
        if "trained_model" in self.app_data:
            del self.app_data["trained_model"]
        if "model_metrics" in self.app_data:
            del self.app_data["model_metrics"]
        if "model_type" in self.app_data:
            del self.app_data["model_type"]
        
        # Disable delete button
        self.delete_btn.setEnabled(False)
    
    def delete_experiment(self):
        """Delete selected experiment"""
        experiment_id = self.exp_combo.currentData()
        if not experiment_id:
            return
        
        try:
            # Confirm deletion
            response = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete experiment {experiment_id}?")
            if response != QMessageBox.Yes:
                return
            
            # Delete experiment
            self.experiment_manager.delete_experiment(experiment_id)
            
            # Update UI
            self.clear_model_info()
            self.exp_combo.removeItem(self.exp_combo.currentIndex())
            self.exp_combo.setCurrentIndex(0)
            self.status_label.setText(f"Experiment {experiment_id} deleted successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting experiment: {str(e)}")
            print(f"Error deleting experiment: {str(e)}")
    
    def browse_files(self):
        """Browse for a data file to predict"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.file_path.setText(file_path)
            
            # Enable prediction if we have a model
            if self.exp_combo.currentIndex() > 0:
                self.predict_btn.setEnabled(True)
            
            # Update status
            self.status_label.setText(f"Selected file: {os.path.basename(file_path)}")
    
    def predict(self):
        """Run prediction on new data"""
        if not self.app_data.get("trained_model"):
            QMessageBox.warning(self, "Warning", "No trained model loaded. Please select an experiment with a trained model.")
            return
        
        file_path = self.file_path.text()
        if file_path == "No file selected":
            QMessageBox.warning(self, "Warning", "No data file selected. Please select a file for prediction.")
            return
        
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.status_label.setText("Loading data...")
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            self.progress_bar.setValue(20)
            self.status_label.setText("Processing data...")
            
            # Check if this is the special voltammetry format (columns are voltage values)
            is_voltammetry_format = False
            voltage_columns = []
            
            # Check if column headers can be converted to float or represent voltage values
            for col in df.columns:
                try:
                    # Try to convert to float directly
                    try:
                        float(col)
                        voltage_columns.append(col)
                        is_voltammetry_format = True
                    except ValueError:
                        # Check if it's a complex voltage format like "-0.795.1925"
                        if isinstance(col, str) and col.replace('-', '').replace('.', '').isdigit():
                            # It has only digits, minus signs, and dots - likely a voltage
                            parts = col.split('.')
                            if len(parts) >= 2:
                                # Try to parse as "major.minor" format
                                try:
                                    voltage_value = float(f"{parts[0]}.{parts[1]}")
                                    voltage_columns.append(col)
                                    is_voltammetry_format = True
                                except ValueError:
                                    pass
                except (TypeError):
                    pass
            
            # Transform data if needed
            if is_voltammetry_format:
                # Transform wide format to long format
                self.status_label.setText("Transforming voltammetry data...")
                
                # Create a long-format dataframe for voltammetric data
                rows = []
                for idx, row in df.iterrows():
                    sample_id = f"Sample_{idx+1}"
                    
                    # Extract metadata columns (non-voltage columns)
                    metadata = {col: row[col] for col in df.columns if col not in voltage_columns}
                    
                    # Add voltage and current data points
                    for voltage_col in voltage_columns:
                        voltage = float(voltage_col.split('.')[0] + '.' + voltage_col.split('.')[1]) if '.' in voltage_col else float(voltage_col)
                        current = row[voltage_col]
                        
                        row_data = {
                            'SampleID': sample_id,
                            'RowIndex': idx,
                            'Potential': voltage,
                            'Current': current
                        }
                        
                        # Add metadata
                        row_data.update(metadata)
                        
                        rows.append(row_data)
                
                # Create a dataframe from the rows
                transformed_df = pd.DataFrame(rows)
                
                # Store the original wide format data
                original_wide_data = df.copy()
                
                # Use the transformed data for further processing
                df = transformed_df
            
            self.progress_bar.setValue(40)
            
            # Get experiment ID and try to load the pipeline
            experiment_id = self.app_data["current_experiment_id"]
            
            # Check if there's a pipeline available for this experiment
            pipeline = self.experiment_manager.load_pipeline(experiment_id)
            
            if pipeline:
                # If we have a pipeline, use that for prediction
                self.status_label.setText("Using pipeline for preprocessing, feature extraction, and prediction...")
                
                # Pipeline will handle preprocessing, feature extraction, and prediction in one step
                self.progress_bar.setValue(80)
                
                # Make predictions using the pipeline
                predictions = pipeline.predict(df)
                
                # Get metrics to determine result format
                metrics = self.app_data.get("model_metrics", {})
                target_col = metrics.get('target_column')
                
                # Get feature columns to display results properly
                feature_columns = metrics.get("feature_columns", [])
                
                # Create a result dataframe
                # First, let's get the feature matrix from the pipeline without the final prediction step
                pipeline_without_model = pipeline[:-1]  # All steps except the last one (model)
                feature_matrix = pipeline_without_model.transform(df)
                
                # Check what type of feature matrix we have
                if isinstance(feature_matrix, pd.DataFrame):
                    # Already a DataFrame
                    result_df = feature_matrix.copy()
                else:
                    # Need to convert to DataFrame with proper column names
                    if feature_columns:
                        result_df = pd.DataFrame(feature_matrix, columns=feature_columns)
                    else:
                        # If we don't have feature column names, create generic ones
                        result_df = pd.DataFrame(feature_matrix)
                
                # Add predictions to the result dataframe
                result_df['Predicted'] = predictions
                
                # No actual values in this case
                has_actual = False
                
                # Display results
                self.display_results(result_df, has_actual, target_col)
                
            else:
                # Fall back to the old approach if no pipeline is available
                self.status_label.setText("No pipeline found. Using separate preprocessing and feature extraction steps...")
                
                # Apply preprocessing if needed
                transformers = self.experiment_manager.get_preprocessing_transformers(experiment_id)
                
                if transformers:
                    self.status_label.setText("Applying preprocessing transformers...")
                    df = apply_transformers(df, transformers)
                
                self.progress_bar.setValue(60)
                
                # Extract features using the same method as the original experiment
                metrics = self.app_data.get("model_metrics", {})
                feature_columns = metrics.get("feature_columns", [])
                
                # Determine feature extraction method based on feature column names
                if any(col.startswith('PC') for col in feature_columns):
                    self.status_label.setText("Extracting features using PCA...")
                    feature_matrix, _ = apply_pca_to_wide_data(df, n_components=3)
                elif any(col.startswith('FFT') for col in feature_columns):
                    self.status_label.setText("Extracting features using FFT...")
                    feature_matrix = apply_fft_to_wide_data(df)
                elif any(col.startswith('Window') for col in feature_columns):
                    self.status_label.setText("Extracting features using Windowed Integral...")
                    feature_matrix = apply_windowed_integral_to_wide_data(df)
                else:
                    # If we can't determine the method, try all methods
                    self.status_label.setText("Trying multiple feature extraction methods...")
                    try:
                        feature_matrix, _ = apply_pca_to_wide_data(df, n_components=3)
                    except:
                        try:
                            feature_matrix = apply_fft_to_wide_data(df)
                        except:
                            try:
                                feature_matrix = apply_windowed_integral_to_wide_data(df)
                            except:
                                raise ValueError("Could not extract features using any method")
                
                self.progress_bar.setValue(80)
                
                # Make predictions
                self.status_label.setText("Making predictions...")
                
                # Get the model
                model = self.app_data["trained_model"]
                
                # Prepare features for prediction
                X = feature_matrix[feature_columns].values
                
                # Make predictions
                predictions = model.predict(X)
                
                # Store predictions in feature matrix
                feature_matrix['Predicted'] = predictions
                
                # Check if target column exists in the data
                target_col = metrics.get('target_column')
                has_actual = target_col in feature_matrix.columns
                
                # Display results
                self.display_results(feature_matrix, has_actual, target_col)
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Prediction completed successfully!")
            
            # Hide progress bar after a delay
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error during prediction: {str(e)}")
    
    def display_results(self, results_df, has_actual, target_col):
        """Display prediction results in table and visualization"""
        try:
            # Clear existing results
            self.results_table.setRowCount(0)
            
            # Set up table
            if has_actual:
                self.results_table.setColumnCount(3)
                self.results_table.setHorizontalHeaderLabels(["Sample ID", f"Actual {target_col}", "Predicted"])
            else:
                self.results_table.setColumnCount(2)
                self.results_table.setHorizontalHeaderLabels(["Sample ID", "Predicted"])
            
            # Add results to table
            sample_ids = results_df.index if 'SampleID' not in results_df.columns else results_df['SampleID']
            
            for i, (idx, row) in enumerate(results_df.iterrows()):
                self.results_table.insertRow(i)
                
                # Sample ID
                sample_id = sample_ids[i] if isinstance(sample_ids, pd.Series) else f"Sample {i+1}"
                self.results_table.setItem(i, 0, QTableWidgetItem(str(sample_id)))
                
                col_offset = 0
                # Actual value if available
                if has_actual:
                    actual_value = row[target_col]
                    self.results_table.setItem(i, 1, QTableWidgetItem(str(actual_value)))
                    col_offset = 1
                
                # Predicted value
                predicted_value = row['Predicted']
                self.results_table.setItem(i, 1 + col_offset, QTableWidgetItem(str(predicted_value)))
            
            # Resize columns to content
            self.results_table.resizeColumnsToContents()
            
            # Create visualization
            self.figure.clear()
            
            # Get model type from metrics
            metrics = self.app_data.get("model_metrics", {})
            is_regression = metrics.get('is_regression', False)
            
            if is_regression and has_actual:
                # Scatter plot for regression
                ax = self.figure.add_subplot(111)
                ax.scatter(results_df[target_col], results_df['Predicted'], alpha=0.7)
                
                # Add diagonal line (perfect predictions)
                min_val = min(results_df[target_col].min(), results_df['Predicted'].min())
                max_val = max(results_df[target_col].max(), results_df['Predicted'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                ax.set_xlabel(f'Actual {target_col}')
                ax.set_ylabel(f'Predicted {target_col}')
                ax.set_title('Actual vs Predicted Values')
                
                # Add R² value if available
                if 'r2' in metrics:
                    try:
                        r2 = float(metrics['r2'])
                        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert R² value to float: {metrics['r2']}")
            
            elif not is_regression and has_actual:
                # Confusion matrix for classification
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                ax = self.figure.add_subplot(111)
                
                try:
                    # Get unique classes
                    classes = sorted(pd.unique(results_df[target_col]))
                    
                    # Ensure consistent data types for classes
                    # Convert all classes to string to avoid comparison between different types
                    results_df[target_col] = results_df[target_col].astype(str)
                    results_df['Predicted'] = results_df['Predicted'].astype(str)
                    classes = [str(c) for c in classes]
                    
                    # Compute confusion matrix
                    cm = confusion_matrix(results_df[target_col], results_df['Predicted'], labels=classes)
                    
                    # Plot confusion matrix
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
                    
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                except Exception as e:
                    print(f"Error creating confusion matrix: {e}")
                    # Fallback to simple bar chart
                    ax.text(0.5, 0.5, f"Could not create confusion matrix: {str(e)}", 
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Error in Visualization')
            
            else:
                # Bar chart of predictions
                ax = self.figure.add_subplot(111)
                
                if is_regression:
                    # Histogram for regression predictions
                    ax.hist(results_df['Predicted'], bins=10, alpha=0.7)
                    ax.set_xlabel('Predicted Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predictions')
                else:
                    # Bar chart for classification predictions
                    prediction_counts = results_df['Predicted'].value_counts()
                    ax.bar(prediction_counts.index, prediction_counts.values)
                    ax.set_xlabel('Predicted Class')
                    ax.set_ylabel('Count')
                    ax.set_title('Prediction Results')
                    ax.tick_params(axis='x', rotation=45)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Visualization Error", f"Error creating visualization: {str(e)}")
            print(f"Error in display_results: {str(e)}")
    
    def go_to_data_import_tab(self):
        """Navigate to the Data Import tab safely"""
        try:
            # Try to access tabs through parent
            self.parent().tabs.setCurrentIndex(0)
        except Exception as e:
            print(f"Warning: Could not navigate to Data Import tab: {e}")
            QMessageBox.information(
                self, 
                "Navigation", 
                "To create a new experiment, please go to the Data Import tab."
            )
    
    def closeEvent(self, event):
        """Clean up when the widget is closed"""
        if hasattr(self, 'experiment_manager') and self.experiment_manager:
            self.experiment_manager.close()
        super().closeEvent(event)
