from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QGridLayout, QTextEdit,
                             QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime


class OptimizationResultsTab(QWidget):
    """
    Tab for displaying optimization results and generating reports
    """
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Summary Section
        summary_group = QGroupBox("Optimization Summary")
        summary_layout = QGridLayout()
        
        # Parameter info
        summary_layout.addWidget(QLabel("Parameter:"), 0, 0)
        self.param_label = QLabel("N/A")
        summary_layout.addWidget(self.param_label, 0, 1)
        
        # Model info
        summary_layout.addWidget(QLabel("Best Model:"), 1, 0)
        self.model_label = QLabel("N/A")
        summary_layout.addWidget(self.model_label, 1, 1)
        
        # Optimal value
        summary_layout.addWidget(QLabel("Optimal Value:"), 2, 0)
        self.optimal_label = QLabel("N/A")
        summary_layout.addWidget(self.optimal_label, 2, 1)
        
        # R² value
        summary_layout.addWidget(QLabel("R² Value:"), 0, 2)
        self.r2_label = QLabel("N/A")
        summary_layout.addWidget(self.r2_label, 0, 3)
        
        # MSE value
        summary_layout.addWidget(QLabel("MSE Value:"), 1, 2)
        self.mse_label = QLabel("N/A")
        summary_layout.addWidget(self.mse_label, 1, 3)
        
        # Equation
        summary_layout.addWidget(QLabel("Equation:"), 2, 2)
        self.equation_label = QLabel("N/A")
        summary_layout.addWidget(self.equation_label, 2, 3)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Visualization Section
        viz_group = QGroupBox("Result Visualization")
        viz_layout = QVBoxLayout()
        
        # Plot for model visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Report Section
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout()
        
        # Report preview
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        report_layout.addWidget(self.report_preview)
        
        # Buttons for report generation
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Report")
        self.generate_btn.clicked.connect(self.generate_report)
        btn_layout.addWidget(self.generate_btn)
        
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        btn_layout.addWidget(self.export_btn)
        
        report_layout.addLayout(btn_layout)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        self.setLayout(layout)
    
    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)
        self.update_results()
    
    def update_results(self):
        """Update the results display"""
        # Check if model is available
        if "regression_model" not in self.app_data:
            return
        
        # Get the model result
        model_result = self.app_data["regression_model"]
        param_data = self.app_data["parameter_data"]
        
        # Update summary labels
        self.param_label.setText(param_data["param_col"])
        self.model_label.setText(model_result["type"].capitalize())
        self.optimal_label.setText(f"{model_result['optimal_x']:.4f}")
        self.r2_label.setText(f"{model_result['metrics']['r2']:.4f}")
        self.mse_label.setText(f"{model_result['metrics']['mse']:.4f}")
        self.equation_label.setText(model_result["equation"])
        
        # Update the plot
        self.update_plot(param_data, model_result)
        
        # Generate report preview
        self.generate_report()
    
    def update_plot(self, param_data, model_result):
        """Update the plot with the fitted model"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get data
        x = param_data["param_values"]
        y = param_data["response_values"]
        
        # Plot the original data
        ax.scatter(x, y, color='blue', label='Data Points')
        
        # Sort x for smooth curve plotting
        x_sorted = np.sort(x)
        x_fine = np.linspace(min(x), max(x), 100)  # For smoother curve
        
        # Generate predictions based on model type
        if model_result["type"] in ["linear", "polynomial2", "polynomial3", "symbolic"]:
            model = model_result["model"]
            y_pred = model.predict(x_fine.reshape(-1, 1))
        else:
            # For gaussian, sigmoid models that use function and parameters
            model_func = model_result["model"]
            if "model_params" in model_result:
                y_pred = model_func(x_fine, *model_result["model_params"])
            else:
                # Fallback to params if model_params doesn't exist
                y_pred = model_func(x_fine, *[model_result["params"][p] for p in model_result["params"]])
        
        # Plot the model curve
        ax.plot(x_fine, y_pred, 'r-', label=f'{model_result["type"].capitalize()} Model')
        
        # Mark the optimal parameter value
        optimal_x = model_result["optimal_x"]
        
        # Calculate the corresponding y value for the optimal x
        if model_result["type"] in ["linear", "polynomial2", "polynomial3", "symbolic"]:
            optimal_y = model_result["model"].predict(np.array([[optimal_x]]))[0]
        else:
            if "model_params" in model_result:
                optimal_y = model_result["model"](optimal_x, *model_result["model_params"])
            else:
                # Fallback to params if model_params doesn't exist
                optimal_y = model_result["model"](optimal_x, *[model_result["params"][p] for p in model_result["params"]])
        
        ax.plot(optimal_x, optimal_y, 'go', markersize=10, label='Optimal Value')
        ax.axvline(x=optimal_x, color='g', linestyle='--', alpha=0.5)
        
        # Set labels and title
        param_col = param_data["param_col"]
        response_col = param_data["response_col"]
        ax.set_xlabel(param_col)
        ax.set_ylabel(response_col)
        ax.set_title(f"Optimal {param_col} for Maximum {response_col}")
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Update the canvas
        self.canvas.draw()
    
    def generate_report(self):
        """Generate a report of the optimization results"""
        # Check if model is available
        if "regression_model" not in self.app_data:
            self.report_preview.setPlainText("No model results available. Please fit a model first.")
            return
        
        # Get the model result
        model_result = self.app_data["regression_model"]
        param_data = self.app_data["parameter_data"]
        
        # Generate report text
        report_text = "# Parameter Optimization Report\n\n"
        
        # Add date and time
        report_text += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add parameter information
        report_text += "## Parameter Information\n\n"
        report_text += f"**Parameter:** {param_data['param_col']}\n"
        report_text += f"**Response Variable:** {param_data['response_col']}\n"
        report_text += f"**Number of Data Points:** {len(param_data['param_values'])}\n\n"
        
        # Add model information
        report_text += "## Model Information\n\n"
        report_text += f"**Model Type:** {model_result['type'].capitalize()}\n"
        report_text += f"**Equation:** {model_result['equation']}\n"
        report_text += f"**R² Value:** {model_result['metrics']['r2']:.4f}\n"
        report_text += f"**MSE Value:** {model_result['metrics']['mse']:.4f}\n\n"
        
        # Add optimization results
        report_text += "## Optimization Results\n\n"
        report_text += f"**Optimal {param_data['param_col']}:** {model_result['optimal_x']:.4f}\n"
        
        # Add model parameters
        report_text += "\n## Model Parameters\n\n"
        if model_result["type"] == "linear":
            report_text += f"**Slope:** {model_result['params']['slope']:.4f}\n"
            report_text += f"**Intercept:** {model_result['params']['intercept']:.4f}\n"
        elif model_result["type"] == "polynomial2" or model_result["type"] == "polynomial3":
            coefficients = model_result['params']['coefficients']
            for i, coef in enumerate(coefficients):
                report_text += f"**Coefficient {i}:** {coef:.6f}\n"
            report_text += f"**Intercept:** {model_result['params']['intercept']:.6f}\n"
        elif model_result["type"] == "gaussian":
            report_text += f"**Amplitude (a):** {model_result['params']['a']:.4f}\n"
            report_text += f"**Mean (b):** {model_result['params']['b']:.4f}\n"
            report_text += f"**Standard Deviation (c):** {model_result['params']['c']:.4f}\n"
        elif model_result["type"] == "sigmoid":
            report_text += f"**Amplitude (a):** {model_result['params']['a']:.4f}\n"
            report_text += f"**Steepness (b):** {model_result['params']['b']:.4f}\n"
            report_text += f"**Midpoint (c):** {model_result['params']['c']:.4f}\n"
            report_text += f"**Offset (d):** {model_result['params']['d']:.4f}\n"
        elif model_result["type"] == "symbolic":
            report_text += f"**Symbolic Expression:** {model_result['equation']}\n"
            # Symbolic regression doesn't usually have explicit parameters to show
        
        # Add conclusion
        report_text += "\n## Conclusion\n\n"
        report_text += f"Based on the {model_result['type']} regression model, the optimal value for "
        report_text += f"{param_data['param_col']} is **{model_result['optimal_x']:.4f}**. "
        report_text += f"This value is expected to maximize the {param_data['response_col']} response. "
        report_text += f"The model has an R² value of {model_result['metrics']['r2']:.4f}, "
        
        if model_result['metrics']['r2'] > 0.9:
            report_text += "indicating an excellent fit to the data."
        elif model_result['metrics']['r2'] > 0.7:
            report_text += "indicating a good fit to the data."
        elif model_result['metrics']['r2'] > 0.5:
            report_text += "indicating a moderate fit to the data."
        else:
            report_text += "indicating a poor fit to the data. Consider collecting more data or trying a different model."
        
        # Display the report
        self.report_preview.setMarkdown(report_text)
        
        # Store the report in app_data
        self.app_data["optimization_report"] = report_text
    
    def export_report(self):
        """Export the report to a file"""
        # Check if report is available
        if "optimization_report" not in self.app_data:
            QMessageBox.warning(self, "Warning", "Please generate a report first")
            return
        
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Write report to file
            with open(file_path, 'w') as f:
                f.write(self.app_data["optimization_report"])
            
            QMessageBox.information(self, "Success", f"Report saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving report: {str(e)}")
