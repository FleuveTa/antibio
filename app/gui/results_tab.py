from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QCheckBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QSplitter,
                             QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from collections import Counter


class ResultsVisualizationTab(QWidget):
    report_ready = pyqtSignal(bool)
    analysis_done = pyqtSignal(bool)  # New signal for enabling prediction tab

    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)

        # Model Evaluation Section
        eval_group = QGroupBox("Model Evaluation")
        eval_layout = QVBoxLayout()

        # Create tabs for different visualizations
        eval_tabs = QTabWidget()

        # Model Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)

        # Model information section
        model_info = QGroupBox("Model Information")
        model_info_layout = QGridLayout()

        model_info_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_label = QLabel("Not available")
        model_info_layout.addWidget(self.model_type_label, 0, 1)

        model_info_layout.addWidget(QLabel("Task Type:"), 1, 0)
        self.task_type_label = QLabel("Not available")
        model_info_layout.addWidget(self.task_type_label, 1, 1)

        model_info_layout.addWidget(QLabel("Target Variable:"), 2, 0)
        self.target_label = QLabel("Not available")
        model_info_layout.addWidget(self.target_label, 2, 1)

        model_info_layout.addWidget(QLabel("Test Size:"), 3, 0)
        self.test_size_label = QLabel("Not available")
        model_info_layout.addWidget(self.test_size_label, 3, 1)

        model_info.setLayout(model_info_layout)
        summary_layout.addWidget(model_info)

        # Metrics section
        metrics_info = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)

        metrics_info.setLayout(metrics_layout)
        summary_layout.addWidget(metrics_info)

        eval_tabs.addTab(summary_tab, "Model Summary")

        # Prediction Results tab
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)

        # Prediction visualization
        self.prediction_figure = Figure(figsize=(8, 6), dpi=100)
        self.prediction_canvas = FigureCanvas(self.prediction_figure)

        # Add navigation toolbar for zoom/pan functionality
        self.prediction_toolbar = NavigationToolbar(self.prediction_canvas, self)

        prediction_layout.addWidget(self.prediction_toolbar)
        prediction_layout.addWidget(self.prediction_canvas)

        eval_tabs.addTab(prediction_tab, "Prediction Results")

        # Feature Importance tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)

        # Feature importance visualization
        self.feature_figure = Figure(figsize=(8, 6), dpi=100)
        self.feature_canvas = FigureCanvas(self.feature_figure)

        # Add navigation toolbar for zoom/pan functionality
        self.feature_toolbar = NavigationToolbar(self.feature_canvas, self)

        feature_layout.addWidget(self.feature_toolbar)
        feature_layout.addWidget(self.feature_canvas)

        eval_tabs.addTab(feature_tab, "Feature Importance")

        # Confusion Matrix tab (for classification)
        confusion_tab = QWidget()
        confusion_layout = QVBoxLayout(confusion_tab)

        # Confusion matrix visualization
        self.confusion_figure = Figure(figsize=(8, 6), dpi=100)
        self.confusion_canvas = FigureCanvas(self.confusion_figure)

        # Add navigation toolbar for zoom/pan functionality
        self.confusion_toolbar = NavigationToolbar(self.confusion_canvas, self)

        confusion_layout.addWidget(self.confusion_toolbar)
        confusion_layout.addWidget(self.confusion_canvas)

        eval_tabs.addTab(confusion_tab, "Confusion Matrix")

        eval_layout.addWidget(eval_tabs)
        eval_group.setLayout(eval_layout)
        splitter.addWidget(eval_group)

        # Report Generation Section
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout()

        # Report options
        options_layout = QGridLayout()

        self.include_summary_check = QCheckBox("Include Model Summary")
        self.include_summary_check.setChecked(True)
        options_layout.addWidget(self.include_summary_check, 0, 0)

        self.include_pred_check = QCheckBox("Include Prediction Results")
        self.include_pred_check.setChecked(True)
        options_layout.addWidget(self.include_pred_check, 0, 1)

        self.include_feature_check = QCheckBox("Include Feature Importance")
        self.include_feature_check.setChecked(True)
        options_layout.addWidget(self.include_feature_check, 1, 0)

        self.include_cm_check = QCheckBox("Include Confusion Matrix")
        self.include_cm_check.setChecked(True)
        options_layout.addWidget(self.include_cm_check, 1, 1)

        report_layout.addLayout(options_layout)

        # Report preview
        report_layout.addWidget(QLabel("Report Preview:"))
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setMinimumHeight(150)
        report_layout.addWidget(self.report_preview)

        # Generate and export buttons
        buttons_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Generate Preview")
        self.preview_btn.clicked.connect(self.generate_report_preview)
        buttons_layout.addWidget(self.preview_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        buttons_layout.addWidget(self.export_btn)

        # Add a button to continue to prediction
        self.continue_btn = QPushButton("Continue to Prediction")
        self.continue_btn.setStyleSheet("background-color: #e6f2ff; font-weight: bold;")
        self.continue_btn.clicked.connect(self.continue_to_prediction)
        buttons_layout.addWidget(self.continue_btn)

        report_layout.addLayout(buttons_layout)

        report_group.setLayout(report_layout)
        splitter.addWidget(report_group)

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Connect signals
        self.report_ready.connect(lambda ready: self.export_btn.setEnabled(ready))

    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)

        # Update displays with current model results
        self.update_model_summary()
        self.update_prediction_plot()
        self.update_feature_importance_plot()
        self.update_confusion_matrix_plot()

    def update_model_summary(self):
        """Update the model summary information"""
        # Check if model results are available
        if "model_results" not in self.app_data:
            return

        results = self.app_data["model_results"]
        model_config = self.app_data.get("model_config", {})

        # Update model information
        self.model_type_label.setText(model_config.get("model_type", "Unknown"))

        is_regression = results.get("is_regression", False)
        self.task_type_label.setText("Regression" if is_regression else "Classification")

        target_variable = self.app_data.get("target_variable", "Unknown")
        self.target_label.setText(target_variable)

        test_size = model_config.get("test_size", 0.2)
        self.test_size_label.setText(f"{test_size:.2f} ({int(test_size*100)}%)")

        # Update metrics table
        metrics = results.get("metrics", {})
        self.metrics_table.setRowCount(len(metrics))

        for i, (metric, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric.upper()))

            # Format value based on type
            if isinstance(value, float):
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            elif isinstance(value, (list, np.ndarray)):
                # For confusion matrix or other array values
                if metric == "confusion_matrix":
                    # Skip confusion matrix in table - it's shown in its own tab
                    self.metrics_table.setItem(i, 1, QTableWidgetItem("See Confusion Matrix tab"))
                else:
                    self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))
            else:
                self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Do not emit signal that analysis is done here
        # This will be done when the user clicks the "Continue to Prediction" button

    def update_prediction_plot(self):
        """Update the prediction results plot"""
        self.prediction_figure.clear()
        ax = self.prediction_figure.add_subplot(111)

        # Check if model results are available
        if "model_results" not in self.app_data:
            ax.text(0.5, 0.5, "No model results available", ha="center", va="center")
            self.prediction_canvas.draw()
            return

        results = self.app_data["model_results"]
        is_regression = results.get("is_regression", False)

        # Get test data and predictions
        y_test = results.get("y_test")
        y_pred = results.get("y_pred")

        if y_test is None or y_pred is None or len(y_test) == 0 or len(y_pred) == 0:
            ax.text(0.5, 0.5, "No prediction data available", ha="center", va="center")
            self.prediction_canvas.draw()
            return

        # Plot based on task type
        if is_regression:
            # Scatter plot of actual vs predicted values
            ax.scatter(y_test, y_pred, alpha=0.7)

            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            # Add labels
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')

            # Add R² value
            r2 = results.get("metrics", {}).get("r2", 0)
            ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # For classification, show bar chart of actual vs predicted class counts
            actual_counts = Counter(y_test)
            pred_counts = Counter(y_pred)

            # Get all unique classes
            all_classes = sorted(list(set(actual_counts.keys()) | set(pred_counts.keys())))

            # Create bar chart
            x = np.arange(len(all_classes))
            width = 0.35

            actual_values = [actual_counts.get(cls, 0) for cls in all_classes]
            pred_values = [pred_counts.get(cls, 0) for cls in all_classes]

            ax.bar(x - width/2, actual_values, width, label='Actual')
            ax.bar(x + width/2, pred_values, width, label='Predicted')

            # Add labels
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Actual vs Predicted Class Counts')
            ax.set_xticks(x)
            ax.set_xticklabels(all_classes)
            ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        self.prediction_canvas.draw()

    def update_feature_importance_plot(self):
        """Update the feature importance plot"""
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        # Check if model results are available
        if "model_results" not in self.app_data:
            ax.text(0.5, 0.5, "No model results available", ha="center", va="center")
            self.feature_canvas.draw()
            return

        results = self.app_data["model_results"]
        feature_importances = results.get("feature_importances")

        if feature_importances is None or len(feature_importances) == 0:
            ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")
            self.feature_canvas.draw()
            return

        # Get feature names
        feature_names = self.app_data.get("selected_features", [])

        # If feature names not available, use generic names
        if not feature_names or len(feature_names) != len(feature_importances):
            feature_names = [f"Feature {i+1}" for i in range(len(feature_importances))]

        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[indices]
        sorted_names = [feature_names[i] for i in indices]

        # Limit to top 20 features for readability
        if len(sorted_names) > 20:
            sorted_names = sorted_names[:20]
            sorted_importances = sorted_importances[:20]

        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')

        self.feature_canvas.draw()

    def update_confusion_matrix_plot(self):
        """Update the confusion matrix plot"""
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)

        # Check if model results are available
        if "model_results" not in self.app_data:
            ax.text(0.5, 0.5, "No model results available", ha="center", va="center")
            self.confusion_canvas.draw()
            return

        results = self.app_data["model_results"]
        is_regression = results.get("is_regression", False)

        # Only show confusion matrix for classification tasks
        if is_regression:
            ax.text(0.5, 0.5, "Confusion matrix not applicable for regression tasks",
                   ha="center", va="center")
            self.confusion_canvas.draw()
            return

        # Get confusion matrix from results
        cm = results.get("metrics", {}).get("confusion_matrix")

        if cm is None:
            # If not available, compute it from y_test and y_pred
            y_test = results.get("y_test")
            y_pred = results.get("y_pred")

            if y_test is None or y_pred is None or len(y_test) == 0 or len(y_pred) == 0:
                ax.text(0.5, 0.5, "No data available to compute confusion matrix",
                       ha="center", va="center")
                self.confusion_canvas.draw()
                return

            # Compute confusion matrix
            cm = sk_confusion_matrix(y_test, y_pred)

        # Convert to numpy array if it's a list
        if isinstance(cm, list):
            cm = np.array(cm)

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Get class labels
        classes = sorted(list(set(results.get("y_test", []))))
        if not classes:
            classes = [f"Class {i}" for i in range(len(cm))]  # Fallback

        # Add labels
        ax.set(xticks=np.arange(len(classes)),
               yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label',
               title='Confusion Matrix')

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        # Calculate threshold safely
        if hasattr(cm, 'max'):
            thresh = cm.max() / 2.
        else:
            # Fallback if max() is not available
            thresh = np.max(cm) / 2. if cm.size > 0 else 0

        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        self.confusion_canvas.draw()

    def generate_report_preview(self):
        """Generate a preview of the report"""
        report_text = "# Model Evaluation Report\n\n"

        # Add date and time
        report_text += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Check if model results are available
        if "model_results" not in self.app_data:
            report_text += "No model results available.\n"
            self.report_preview.setText(report_text)
            return

        results = self.app_data["model_results"]
        model_config = self.app_data.get("model_config", {})
        is_regression = results.get("is_regression", False)

        # Add model summary if selected
        if self.include_summary_check.isChecked():
            report_text += "## Model Summary\n\n"
            report_text += f"**Model Type:** {model_config.get('model_type', 'Unknown')}\n"
            report_text += f"**Task Type:** {'Regression' if is_regression else 'Classification'}\n"
            report_text += f"**Target Variable:** {self.app_data.get('target_variable', 'Unknown')}\n"
            report_text += f"**Test Size:** {model_config.get('test_size', 0.2):.2f}\n\n"

            # Add metrics
            metrics = results.get("metrics", {})
            if metrics:
                report_text += "### Performance Metrics\n\n"
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_text += f"- **{metric.upper()}:** {value:.4f}\n"
                    elif not isinstance(value, (list, np.ndarray)):  # Skip arrays/matrices
                        report_text += f"- **{metric.upper()}:** {value}\n"
                report_text += "\n"

        # Add prediction results if selected
        if self.include_pred_check.isChecked():
            report_text += "## Prediction Results\n\n"
            if is_regression:
                report_text += "Actual vs Predicted values plot included in the exported report.\n\n"
            else:
                report_text += "Class distribution plot included in the exported report.\n\n"

        # Add feature importance if selected
        if self.include_feature_check.isChecked():
            report_text += "## Feature Importance\n\n"
            feature_importances = results.get("feature_importances")
            if feature_importances is not None and len(feature_importances) > 0:
                report_text += "Feature importance plot included in the exported report.\n\n"

                # Get feature names
                feature_names = self.app_data.get("selected_features", [])

                # If feature names not available, use generic names
                if not feature_names or len(feature_names) != len(feature_importances):
                    feature_names = [f"Feature {i+1}" for i in range(len(feature_importances))]

                # Sort features by importance
                indices = np.argsort(feature_importances)[::-1]
                sorted_importances = feature_importances[indices]
                sorted_names = [feature_names[i] for i in indices]

                # List top 5 features
                report_text += "Top 5 most important features:\n\n"
                for i in range(min(5, len(sorted_names))):
                    report_text += f"{i+1}. **{sorted_names[i]}:** {sorted_importances[i]:.4f}\n"
                report_text += "\n"
            else:
                report_text += "No feature importance data available.\n\n"

        # Add confusion matrix if selected
        if self.include_cm_check.isChecked() and not is_regression:
            report_text += "## Confusion Matrix\n\n"
            report_text += "Confusion matrix included in the exported report.\n\n"

        self.report_preview.setText(report_text)
        self.report_ready.emit(True)

    def export_report(self):
        """Export the report to a file"""
        # Generate report preview first
        self.generate_report_preview()

        # Ask for file location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "Text Files (*.txt);;Markdown Files (*.md);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write(self.report_preview.toPlainText())

            QMessageBox.information(self, "Success", f"Report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

    def continue_to_prediction(self):
        """Emit signal to continue to prediction tab"""
        # Emit the signal to enable prediction tab
        self.analysis_done.emit(True)
