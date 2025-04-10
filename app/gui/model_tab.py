from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QGroupBox, QGridLayout, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
                             QTabWidget, QTableWidget, QTableWidgetItem, QSplitter,
                             QRadioButton, QButtonGroup, QSlider, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from app.core.models import AntibioticClassifier


class TrainingThread(QThread):
    """Thread for running model training in background"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    
    def __init__(self, model_config, X, y):
        super().__init__()
        self.model_config = model_config
        self.X = X
        self.y = y
    
    def run(self):
        try:
            # Create classifier with selected model type
            classifier = AntibioticClassifier(model_type=self.model_config['model_type'])
            
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
            
            # Update progress
            self.progress_signal.emit(100)
            
            # Emit results
            self.result_signal.emit(results)
            
        except Exception as e:
            print(f"Error in training thread: {e}")
            self.result_signal.emit({"error": str(e)})


class ModelTrainingTab(QWidget):
    model_ready = pyqtSignal(bool)
    
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
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
        
        # Cross-validation folds
        config_layout.addWidget(QLabel("CV Folds:"), 3, 0)
        self.cv_spin = QSpinBox()
        self.cv_spin.setRange(2, 10)
        self.cv_spin.setValue(5)
        config_layout.addWidget(self.cv_spin, 3, 1)
        
        # Hyperparameter optimization
        self.optimize_check = QCheckBox("Hyperparameter Optimization")
        config_layout.addWidget(self.optimize_check, 4, 0, 1, 2)
        
        # Model specific params section (will be populated based on model type)
        self.params_frame = QFrame()
        self.params_layout = QGridLayout(self.params_frame)
        config_layout.addWidget(self.params_frame, 5, 0, 1, 2)
        
        # Training button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        config_layout.addWidget(self.train_btn, 6, 0, 1, 2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        config_layout.addWidget(self.progress_bar, 7, 0, 1, 2)
        
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
        if self.app_data.get("processed_data") is not None:
            data = self.app_data["processed_data"]
            self.target_combo.clear()
            self.target_combo.addItems(data.columns)
    
    def train_model(self):
        """Train the model with selected configuration"""
        # Validate that we have data and features
        if (self.app_data.get("processed_data") is None or 
            self.app_data.get("selected_features") is None):
            print("Missing data or features for training")
            return
        
        # Get selected target
        target_col = self.target_combo.currentText()
        if not target_col:
            print("No target column selected")
            return
        
        # Prepare data
        data = self.app_data["processed_data"]
        selected_features = self.app_data["selected_features"]
        
        # Check if features and target are in the dataset
        if target_col not in data.columns:
            print(f"Target column {target_col} not found in dataset")
            return
        
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            print(f"Missing features in dataset: {missing_features}")
            return
        
        # Get feature matrix and target vector
        X = data[selected_features].values
        y = data[target_col].values
        
        # Store target variable name
        self.app_data["target_variable"] = target_col
        
        # Configure model
        model_type = self.model_combo.currentText()
        # Extract abbreviation from model type text
        model_type_code = model_type.split("(")[1].strip(")")
        
        model_config = {
            "model_type": model_type_code,
            "test_size": self.test_spin.value() / 100,  # Convert from percentage
            "cv_folds": self.cv_spin.value(),
            "optimize": self.optimize_check.isChecked()
        }
        
        # Add model-specific parameters
        if hasattr(self, "n_estimators_spin"):
            model_config["n_estimators"] = self.n_estimators_spin.value()
        
        if hasattr(self, "max_depth_spin"):
            model_config["max_depth"] = self.max_depth_spin.value()
            
        if hasattr(self, "learning_rate_spin"):
            model_config["learning_rate"] = self.learning_rate_spin.value()
            
        if hasattr(self, "kernel_combo"):
            model_config["kernel"] = self.kernel_combo.currentText()
            
        if hasattr(self, "c_spin"):
            model_config["C"] = self.c_spin.value()
            
        if hasattr(self, "gamma_combo"):
            gamma = self.gamma_combo.currentText()
            # Convert numeric strings to float
            if gamma not in ["scale", "auto"]:
                gamma = float(gamma)
            model_config["gamma"] = gamma
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Store model config
        self.app_data["model_config"] = model_config
        
        # Create and start training thread
        self.training_thread = TrainingThread(model_config, X, y)
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
            return
        
        # Store results
        self.app_data["model_results"] = results
        
        # Update metrics table
        metrics = results.get("metrics", {})
        if metrics:
            self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics.get('accuracy', 0):.4f}"))
            self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
            self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"))
            self.metrics_table.setItem(3, 1, QTableWidgetItem(f"{metrics.get('f1', 0):.4f}"))
            self.metrics_table.setItem(4, 1, QTableWidgetItem(f"{metrics.get('roc_auc', 0):.4f}"))
        
        # Update confusion matrix plot
        self.update_confusion_matrix(results)
        
        # Update ROC curve plot
        self.update_roc_curve(results)
        
        # Update feature importance plot
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
        
        # This is a simplified example - in a real app, you would compute
        # the actual confusion matrix from y_test and y_pred
        # For now, we'll create a mock 2x2 confusion matrix for binary classification
        
        y_test = results.get("y_test", [])
        y_pred = results.get("y_pred", [])
        
        if len(y_test) > 0 and len(y_pred) > 0:
            # Compute confusion matrix - simplified example
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            self.cm_figure.colorbar(im, ax=ax)
            
            # Set labels
            classes = ["Negative", "Positive"]
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
        """Update ROC curve visualization"""
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        
        y_test = results.get("y_test", [])
        y_pred_proba = results.get("y_pred_proba", [])
        
        if len(y_test) > 0 and len(y_pred_proba) > 0:
            # Plot ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        
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
        # Update column list and enable controls based on data
        self.update_column_list()