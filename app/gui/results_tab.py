from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QSlider, QSplitter,
                             QTextEdit, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json


class ResultsVisualizationTab(QWidget):
    report_ready = pyqtSignal(bool)
    
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
        
        # Calibration tab
        calibration_tab = QWidget()
        calibration_layout = QVBoxLayout(calibration_tab)
        
        calibration_options = QHBoxLayout()
        calibration_options.addWidget(QLabel("Antibiotic:"))
        self.antibiotic_combo = QComboBox()
        self.antibiotic_combo.addItems(["Tetracycline", "Penicillin", "Streptomycin", "Chloramphenicol"])
        self.antibiotic_combo.currentIndexChanged.connect(lambda: self.update_calibration_plot())
        calibration_options.addWidget(self.antibiotic_combo)
        
        calibration_options.addWidget(QLabel("Concentration Range:"))
        self.conc_min_spin = QDoubleSpinBox()
        self.conc_min_spin.setRange(0, 1000)
        self.conc_min_spin.setValue(0)
        self.conc_min_spin.setSuffix(" ppb")
        calibration_options.addWidget(self.conc_min_spin)
        
        self.conc_max_spin = QDoubleSpinBox()
        self.conc_max_spin.setRange(1, 10000)
        self.conc_max_spin.setValue(1000)
        self.conc_max_spin.setSuffix(" ppb")
        calibration_options.addWidget(self.conc_max_spin)
        
        self.generate_calibration_btn = QPushButton("Generate Calibration")
        self.generate_calibration_btn.clicked.connect(self.update_calibration_plot)
        calibration_options.addWidget(self.generate_calibration_btn)
        
        calibration_layout.addLayout(calibration_options)
        
        # Calibration plot
        self.calibration_figure = Figure(figsize=(8, 6), dpi=100)
        self.calibration_canvas = FigureCanvas(self.calibration_figure)
        calibration_layout.addWidget(self.calibration_canvas)
        
        # Add calibration information section
        calibration_info = QGroupBox("Calibration Information")
        cal_info_layout = QGridLayout()
        
        cal_info_layout.addWidget(QLabel("Equation:"), 0, 0)
        self.equation_label = QLabel("y = mx + b")
        cal_info_layout.addWidget(self.equation_label, 0, 1)
        
        cal_info_layout.addWidget(QLabel("R² Value:"), 1, 0)
        self.r2_label = QLabel("0.0000")
        cal_info_layout.addWidget(self.r2_label, 1, 1)
        
        cal_info_layout.addWidget(QLabel("Limit of Detection:"), 2, 0)
        self.lod_label = QLabel("0.0 ppb")
        cal_info_layout.addWidget(self.lod_label, 2, 1)
        
        cal_info_layout.addWidget(QLabel("Limit of Quantification:"), 3, 0)
        self.loq_label = QLabel("0.0 ppb")
        cal_info_layout.addWidget(self.loq_label, 3, 1)
        
        cal_info_layout.addWidget(QLabel("Linear Range:"), 4, 0)
        self.range_label = QLabel("0.0 - 0.0 ppb")
        cal_info_layout.addWidget(self.range_label, 4, 1)
        
        calibration_info.setLayout(cal_info_layout)
        calibration_layout.addWidget(calibration_info)
        
        eval_tabs.addTab(calibration_tab, "Calibration Curve")
        
        # Prediction tab
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)
        
        # Sample selection
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Test Sample:"))
        self.sample_combo = QComboBox()
        self.sample_combo.addItems(["Sample 1", "Sample 2", "Sample 3", "Sample 4"])
        self.sample_combo.currentIndexChanged.connect(lambda: self.update_prediction_plot())
        sample_layout.addWidget(self.sample_combo)
        
        self.predict_btn = QPushButton("Predict Concentration")
        self.predict_btn.clicked.connect(self.update_prediction_plot)
        sample_layout.addWidget(self.predict_btn)
        
        prediction_layout.addLayout(sample_layout)
        
        # Prediction results visualization
        self.prediction_figure = Figure(figsize=(8, 6), dpi=100)
        self.prediction_canvas = FigureCanvas(self.prediction_figure)
        prediction_layout.addWidget(self.prediction_canvas)
        
        # Prediction results table
        self.prediction_table = QTableWidget(4, 3)
        self.prediction_table.setHorizontalHeaderLabels(["Antibiotic", "Predicted Conc.", "Actual Conc."])
        self.prediction_table.horizontalHeader().setStretchLastSection(True)
        prediction_layout.addWidget(self.prediction_table)
        
        eval_tabs.addTab(prediction_tab, "Prediction Analysis")
        
        # Parameter Optimization tab
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)
        
        # Parameter selection
        param_control_layout = QHBoxLayout()
        param_control_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "pH", "Temperature", "Scan Rate", "Deposition Time", 
            "Amplitude", "Frequency", "Step Potential"
        ])
        self.param_combo.currentIndexChanged.connect(lambda: self.update_param_plot())
        param_control_layout.addWidget(self.param_combo)
        
        self.optimize_btn = QPushButton("Optimize Parameter")
        self.optimize_btn.clicked.connect(self.update_param_plot)
        param_control_layout.addWidget(self.optimize_btn)
        
        param_layout.addLayout(param_control_layout)
        
        # Parameter optimization plot
        self.param_figure = Figure(figsize=(8, 6), dpi=100)
        self.param_canvas = FigureCanvas(self.param_figure)
        param_layout.addWidget(self.param_canvas)
        
        # Optimal parameters table
        self.param_table = QTableWidget(7, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Optimal Value"])
        for i, param in enumerate([
            "pH", "Temperature", "Scan Rate", "Deposition Time", 
            "Amplitude", "Frequency", "Step Potential"
        ]):
            self.param_table.setItem(i, 0, QTableWidgetItem(param))
            self.param_table.setItem(i, 1, QTableWidgetItem("--"))
        
        self.param_table.horizontalHeader().setStretchLastSection(True)
        param_layout.addWidget(self.param_table)
        
        eval_tabs.addTab(param_tab, "Parameter Optimization")
        
        eval_layout.addWidget(eval_tabs)
        eval_group.setLayout(eval_layout)
        splitter.addWidget(eval_group)
        
        # Report Generation Section
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout()
        
        # Report options
        options_layout = QGridLayout()
        
        self.include_calib_check = QCheckBox("Include Calibration Curves")
        self.include_calib_check.setChecked(True)
        options_layout.addWidget(self.include_calib_check, 0, 0)
        
        self.include_pred_check = QCheckBox("Include Prediction Results")
        self.include_pred_check.setChecked(True)
        options_layout.addWidget(self.include_pred_check, 0, 1)
        
        self.include_param_check = QCheckBox("Include Parameter Optimization")
        self.include_param_check.setChecked(True)
        options_layout.addWidget(self.include_param_check, 1, 0)
        
        self.include_model_check = QCheckBox("Include Model Details")
        self.include_model_check.setChecked(True)
        options_layout.addWidget(self.include_model_check, 1, 1)
        
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
        
        report_layout.addLayout(buttons_layout)
        
        report_group.setLayout(report_layout)
        splitter.addWidget(report_group)
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def update_calibration_plot(self):
        """Update the calibration curve plot"""
        self.calibration_figure.clear()
        ax = self.calibration_figure.add_subplot(111)
        
        # Get selected antibiotic
        antibiotic = self.antibiotic_combo.currentText()
        
        # Get concentration range
        min_conc = self.conc_min_spin.value()
        max_conc = self.conc_max_spin.value()
        
        # Create mock calibration data
        np.random.seed(42)  # For reproducibility
        concentrations = np.linspace(min_conc, max_conc, 10)
        
        # Different slope for each antibiotic (just for demo)
        slopes = {
            "Tetracycline": 0.015,
            "Penicillin": 0.008,
            "Streptomycin": 0.012,
            "Chloramphenicol": 0.02
        }
        
        slope = slopes.get(antibiotic, 0.01)
        intercept = 0.05
        
        # Generate responses with some noise
        responses = slope * concentrations + intercept + np.random.normal(0, 0.02, 10)
        
        # Plot points
        ax.scatter(concentrations, responses, color='blue', label='Measured Data')
        
        # Calculate and plot regression line
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        model = LinearRegression()
        model.fit(concentrations.reshape(-1, 1), responses)
        
        pred_responses = model.predict(concentrations.reshape(-1, 1))
        r2 = r2_score(responses, pred_responses)
        
        # Plot regression line
        x_range = np.linspace(min_conc, max_conc, 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        ax.plot(x_range, y_pred, color='red', label='Regression Line')
        
        # Calculate LOD and LOQ (simplified example)
        # In real application, these would be calculated properly from blank measurements
        blank_std = 0.005  # Standard deviation of blank
        slope = model.coef_[0]
        
        lod = 3.3 * blank_std / slope
        loq = 10 * blank_std / slope
        
        # Add LOD and LOQ lines
        ax.axhline(y=intercept + 3.3 * blank_std, color='green', linestyle='--', label=f'LOD ({lod:.2f} ppb)')
        ax.axhline(y=intercept + 10 * blank_std, color='orange', linestyle='--', label=f'LOQ ({loq:.2f} ppb)')
        
        # Set labels and title
        ax.set_xlabel('Concentration (ppb)')
        ax.set_ylabel('Response (current, μA)')
        ax.set_title(f'Calibration Curve for {antibiotic}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        self.calibration_canvas.draw()
        
        # Update calibration information
        self.equation_label.setText(f"y = {slope:.4f}x + {intercept:.4f}")
        self.r2_label.setText(f"{r2:.4f}")
        self.lod_label.setText(f"{lod:.2f} ppb")
        self.loq_label.setText(f"{loq:.2f} ppb")
        self.range_label.setText(f"{loq:.2f} - {max_conc:.2f} ppb")
    
    def update_prediction_plot(self):
        """Update the prediction analysis plot"""
        self.prediction_figure.clear()
        ax = self.prediction_figure.add_subplot(111)
        
        # Get selected sample
        sample = self.sample_combo.currentText()
        
        # Create mock prediction data for different antibiotics
        antibiotics = ["Tetracycline", "Penicillin", "Streptomycin", "Chloramphenicol"]
        
        # Mock data - would be replaced with actual predictions
        if sample == "Sample 1":
            # Food sample with multiple antibiotics
            prediction = [120, 0, 85, 15]
            actual = [115, 0, 90, 18]
        elif sample == "Sample 2":
            # Clean sample
            prediction = [0, 0, 0, 0]  
            actual = [0, 0, 0, 0]
        elif sample == "Sample 3":
            # Single antibiotic
            prediction = [0, 450, 0, 0]
            actual = [0, 425, 0, 0]
        else:
            # Multiple antibiotics
            prediction = [75, 30, 60, 0]
            actual = [70, 35, 55, 0]
            
        # Plot bar chart
        x = np.arange(len(antibiotics))
        width = 0.35
        
        ax.bar(x - width/2, prediction, width, label='Predicted')
        ax.bar(x + width/2, actual, width, label='Actual')
        
        ax.set_xticks(x)
        ax.set_xticklabels(antibiotics)
        ax.set_ylabel('Concentration (ppb)')
        ax.set_title(f'Antibiotic Concentrations in {sample}')
        ax.legend()
        
        # Add threshold line for regulatory limits
        # MRLs vary by antibiotic and food type
        mrl_values = [100, 50, 200, 10]  # Example MRLs
        
        for i, mrl in enumerate(mrl_values):
            ax.plot([i-width, i+width], [mrl, mrl], 'r--', alpha=0.7)
            ax.text(i, mrl + 5, f'MRL: {mrl}', ha='center', fontsize=8)
        
        self.prediction_canvas.draw()
        
        # Update prediction table
        for i, antibiotic in enumerate(antibiotics):
            self.prediction_table.setItem(i, 0, QTableWidgetItem(antibiotic))
            self.prediction_table.setItem(i, 1, QTableWidgetItem(f"{prediction[i]:.1f} ppb"))
            self.prediction_table.setItem(i, 2, QTableWidgetItem(f"{actual[i]:.1f} ppb"))
            
            # Highlight values exceeding MRL
            if prediction[i] > mrl_values[i]:
                self.prediction_table.item(i, 1).setBackground(Qt.red)
            
            if actual[i] > mrl_values[i]:
                self.prediction_table.item(i, 2).setBackground(Qt.red)
    
    def update_param_plot(self):
        """Update the parameter optimization plot"""
        self.param_figure.clear()
        ax = self.param_figure.add_subplot(111)
        
        # Get selected parameter
        param = self.param_combo.currentText()
        
        # Create mock parameter optimization data
        param_ranges = {
            "pH": (3, 9),
            "Temperature": (15, 45),
            "Scan Rate": (10, 200),
            "Deposition Time": (30, 300),
            "Amplitude": (5, 100),
            "Frequency": (10, 150),
            "Step Potential": (1, 20)
        }
        
        param_units = {
            "pH": "",
            "Temperature": "°C",
            "Scan Rate": "mV/s",
            "Deposition Time": "s",
            "Amplitude": "mV",
            "Frequency": "Hz",
            "Step Potential": "mV"
        }
        
        # Get range for selected parameter
        min_val, max_val = param_ranges.get(param, (0, 100))
        unit = param_units.get(param, "")
        
        # Generate parameter values
        x = np.linspace(min_val, max_val, 20)
        
        # Generate response values with an optimal point
        # Different optimal point for each parameter
        optimal_indices = {
            "pH": 7,
            "Temperature": 10,
            "Scan Rate": 12,
            "Deposition Time": 15,
            "Amplitude": 8,
            "Frequency": 11,
            "Step Potential": 14
        }
        
        optimal_idx = optimal_indices.get(param, 10)
        optimal_val = x[optimal_idx]
        
        # Generate response curve with maximum at optimal value
        y = -((x - optimal_val) ** 2) + 100 + np.random.normal(0, 5, 20)
        
        # Plot parameter optimization curve
        ax.plot(x, y, 'o-', color='blue')
        
        # Mark optimal point
        ax.plot(optimal_val, y[optimal_idx], 'ro', markersize=10)
        
        # Add annotation for optimal value
        ax.annotate(f'Optimal: {optimal_val:.2f} {unit}',
                    xy=(optimal_val, y[optimal_idx]),
                    xytext=(optimal_val, y[optimal_idx] - 20),
                    arrowprops=dict(arrowstyle='->'),
                    ha='center')
        
        # Set labels and title
        ax.set_xlabel(f'{param} {unit}')
        ax.set_ylabel('Sensor Response (a.u.)')
        ax.set_title(f'Optimization of {param}')
        
        self.param_canvas.draw()
        
        # Update optimal parameter in the table
        for i in range(self.param_table.rowCount()):
            param_name = self.param_table.item(i, 0).text()
            if param_name == param:
                # Get unit for this parameter
                unit = param_units.get(param, "")
                self.param_table.setItem(i, 1, QTableWidgetItem(f"{optimal_val:.2f} {unit}"))
    
    def generate_report_preview(self):
        """Generate a preview of the report"""
        report_text = "# Electrochemical Sensor Analysis Report\n\n"
        
        # Add date and time
        from datetime import datetime
        report_text += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add model information if selected
        if self.include_model_check.isChecked() and self.app_data.get("model_config") is not None:
            model_config = self.app_data["model_config"]
            
            report_text += "## Model Information\n\n"
            report_text += f"**Model Type:** {model_config.get('model_type', 'Unknown')}\n"
            
            if self.app_data.get("model_results") is not None:
                metrics = self.app_data["model_results"].get("metrics", {})
                
                report_text += "### Performance Metrics\n\n"
                report_text += f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                report_text += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                report_text += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                report_text += f"- F1 Score: {metrics.get('f1', 0):.4f}\n"
                report_text += f"- ROC AUC: {metrics.get('roc_auc', 0):.4f}\n\n"
        
        # Add calibration information if selected
        if self.include_calib_check.isChecked():
            report_text += "## Calibration Information\n\n"
            
            antibiotic = self.antibiotic_combo.currentText()
            report_text += f"**Antibiotic:** {antibiotic}\n"
            report_text += f"**Equation:** {self.equation_label.text()}\n"
            report_text += f"**R² Value:** {self.r2_label.text()}\n"
            report_text += f"**Limit of Detection (LOD):** {self.lod_label.text()}\n"
            report_text += f"**Limit of Quantification (LOQ):** {self.loq_label.text()}\n"
            report_text += f"**Linear Range:** {self.range_label.text()}\n\n"
        
        # Add prediction results if selected
        if self.include_pred_check.isChecked():
            report_text += "## Prediction Results\n\n"
            
            sample = self.sample_combo.currentText()
            report_text += f"**Sample:** {sample}\n\n"
            
            report_text += "| Antibiotic | Predicted Conc. | Actual Conc. |\n"
            report_text += "|------------|-----------------|-------------|\n"
            
            for i in range(self.prediction_table.rowCount()):
                antibiotic = self.prediction_table.item(i, 0).text()
                pred = self.prediction_table.item(i, 1).text()
                actual = self.prediction_table.item(i, 2).text()
                
                report_text += f"| {antibiotic} | {pred} | {actual} |\n"
            
            report_text += "\n"
        
        # Add parameter optimization if selected
        if self.include_param_check.isChecked():
            report_text += "## Parameter Optimization\n\n"
            
            report_text += "| Parameter | Optimal Value |\n"
            report_text += "|-----------|---------------|\n"
            
            for i in range(self.param_table.rowCount()):
                param = self.param_table.item(i, 0).text()
                value = self.param_table.item(i, 1).text()
                
                report_text += f"| {param} | {value} |\n"
            
            report_text += "\n"
        
        # Add conclusion
        report_text += "## Conclusion\n\n"
        report_text += "This report presents the results of electrochemical sensor analysis for antibiotic detection in food samples. "
        report_text += "The calibration curves, prediction results, and parameter optimization provide insights into the performance "
        report_text += "and applicability of the developed sensor system for routine monitoring of antibiotics in food products.\n\n"
        
        # Display the report preview
        self.report_preview.setMarkdown(report_text)
        
        # Store report text in app_data
        self.app_data["report_text"] = report_text
        
        # Signal that report is ready
        self.report_ready.emit(True)
    
    def export_report(self):
        """Export the report to a file"""
        # Generate report if not already done
        if "report_text" not in self.app_data:
            self.generate_report_preview()
        
        # Ask for save location
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Report", "", "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.app_data["report_text"])
                print(f"Report saved to {file_path}")
            except Exception as e:
                print(f"Error saving report: {e}")
    
    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)
        
        # Generate initial plots
        self.update_calibration_plot()
        self.update_prediction_plot()
        self.update_param_plot()