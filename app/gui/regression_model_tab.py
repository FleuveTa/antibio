from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QGroupBox, QGridLayout,
                             QTableWidget, QTableWidgetItem, QMessageBox,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.core.regression_models import RegressionModels, SYMBOLIC_AVAILABLE


class RegressionModelTab(QWidget):
    """
    Tab for fitting regression models to parameter data
    """
    model_fitted = pyqtSignal(bool)

    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model Selection Section
        model_group = QGroupBox("Regression Model Selection")
        model_layout = QGridLayout()

        # Model type selection
        self.model_group = QButtonGroup(self)

        # Polynomial (Degree 2)
        self.poly2_radio = QRadioButton("Polynomial (Degree 2)")
        self.poly2_radio.setChecked(True)
        self.model_group.addButton(self.poly2_radio, 2)
        model_layout.addWidget(self.poly2_radio, 0, 0)

        # Polynomial (Degree 3)
        self.poly3_radio = QRadioButton("Polynomial (Degree 3)")
        self.model_group.addButton(self.poly3_radio, 3)
        model_layout.addWidget(self.poly3_radio, 0, 1)

        # Gaussian Regression
        self.gaussian_radio = QRadioButton("Gaussian Regression")
        self.model_group.addButton(self.gaussian_radio, 4)
        model_layout.addWidget(self.gaussian_radio, 1, 0)

        # Sigmoid Regression
        self.sigmoid_radio = QRadioButton("Sigmoid Regression")
        self.model_group.addButton(self.sigmoid_radio, 5)
        model_layout.addWidget(self.sigmoid_radio, 1, 1)

        # Symbolic Regression
        self.symbolic_radio = QRadioButton("Symbolic Regression")
        self.symbolic_radio.setEnabled(True)
        self.model_group.addButton(self.symbolic_radio, 6)
        model_layout.addWidget(self.symbolic_radio, 1, 2)

        # Auto option
        self.auto_radio = QRadioButton("Auto (Best Fit)")
        self.model_group.addButton(self.auto_radio, 7)
        model_layout.addWidget(self.auto_radio, 2, 0, 1, 3)

        # Fit button
        self.fit_btn = QPushButton("Fit Model")
        self.fit_btn.clicked.connect(self.fit_model)
        model_layout.addWidget(self.fit_btn, 3, 0, 1, 3)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Model Results Section
        results_group = QGroupBox("Model Results")
        results_layout = QVBoxLayout()

        # Plot for model visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        # Table for model metrics
        metrics_layout = QHBoxLayout()

        # Equation display
        metrics_layout.addWidget(QLabel("Equation:"))
        self.equation_label = QLabel("Not fitted yet")
        metrics_layout.addWidget(self.equation_label)

        metrics_layout.addStretch()

        # R² display
        metrics_layout.addWidget(QLabel("R²:"))
        self.r2_label = QLabel("N/A")
        metrics_layout.addWidget(self.r2_label)

        # MSE display
        metrics_layout.addWidget(QLabel("MSE:"))
        self.mse_label = QLabel("N/A")
        metrics_layout.addWidget(self.mse_label)

        # Optimal parameter display
        metrics_layout.addWidget(QLabel("Optimal Value:"))
        self.optimal_label = QLabel("N/A")
        metrics_layout.addWidget(self.optimal_label)

        results_layout.addLayout(metrics_layout)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def update_plot(self, x, y, model_result):
        """Update the plot with the fitted model"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Plot the original data
        ax.scatter(x, y, color='blue', label='Data Points')

        # Sort x for smooth curve plotting
        x_sorted = np.sort(x)

        # Generate predictions for the sorted x values
        if model_result["type"] in ["polynomial2", "polynomial3", "symbolic"]:
            model = model_result["model"]
            y_pred = model.predict(x_sorted.reshape(-1, 1))
        elif model_result["type"] in ["gaussian", "sigmoid"]:
            model_func = model_result["model"]
            if "model_params" in model_result:
                y_pred = model_func(x_sorted, *model_result["model_params"])
            else:
                # Fallback to params if model_params doesn't exist
                y_pred = model_func(x_sorted, *[model_result["params"][p] for p in model_result["params"]])
        else:
            y_pred = None

        # Plot the model curve
        if y_pred is not None:
            ax.plot(x_sorted, y_pred, 'r-', label=f'{model_result["type"].capitalize()} Model')

        # Mark the optimal parameter value
        optimal_x = model_result["optimal_x"]
        if model_result["type"] in ["polynomial2", "polynomial3", "symbolic"]:
            optimal_y = model_result["model"].predict(np.array([[optimal_x]]))[0]
        elif model_result["type"] in ["gaussian", "sigmoid"]:
            if "model_params" in model_result:
                optimal_y = model_result["model"](optimal_x, *model_result["model_params"])
            else:
                # Fallback to params if model_params doesn't exist
                optimal_y = model_result["model"](optimal_x, *[model_result["params"][p] for p in model_result["params"]])
        else:
            optimal_y = None

        if optimal_y is not None:
            ax.plot(optimal_x, optimal_y, 'go', markersize=10, label='Optimal Value')

        # Set labels and title
        param_col = self.app_data["parameter_data"]["param_col"]
        response_col = self.app_data["parameter_data"]["response_col"]
        ax.set_xlabel(param_col)
        ax.set_ylabel(response_col)
        ax.set_title(f"{model_result['type'].capitalize()} Model: {response_col} vs {param_col}")

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Update the canvas
        self.canvas.draw()

    def fit_model(self):
        """Fit the selected regression model to the data"""
        # Check if data is available
        if "parameter_data" not in self.app_data:
            QMessageBox.warning(self, "Warning", "Please import parameter data first")
            return

        # Get the data
        param_data = self.app_data["parameter_data"]
        x = param_data["param_values"]
        y = param_data["response_values"]

        try:
            # Determine which model to fit
            model_id = self.model_group.checkedId()

            if model_id == 2:  # Polynomial (Degree 2)
                model_result = RegressionModels.polynomial_regression_2(x, y)
                model_type = "polynomial2"
            elif model_id == 3:  # Polynomial (Degree 3)
                model_result = RegressionModels.polynomial_regression_3(x, y)
                model_type = "polynomial3"
            elif model_id == 4:  # Gaussian
                model_result = RegressionModels.gaussian_regression(x, y)
                model_type = "gaussian"
            elif model_id == 5:  # Sigmoid
                model_result = RegressionModels.sigmoid_regression(x, y)
                model_type = "sigmoid"
            elif model_id == 6:  # Symbolic
                model_result = RegressionModels.symbolic_regression(x, y)
                model_type = "symbolic"
            elif model_id == 7:  # Auto
                model_result = RegressionModels.find_best_model(x, y)
                model_type = model_result["type"] if model_result else None

            if model_result is None:
                QMessageBox.warning(self, "Warning", f"Failed to fit {model_type} model to the data")
                return

            # Store the model result
            self.app_data["regression_model"] = model_result

            # Update the plot
            self.update_plot(x, y, model_result)

            # Update the metrics
            self.update_metrics(model_result)

            # Emit signal that model is fitted
            self.model_fitted.emit(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error fitting model: {str(e)}")

    def update_metrics(self, model_result):
        """Update the metrics display with model statistics"""
        if not model_result:
            return
            
        metrics = model_result.get("metrics", {})
        if not metrics and model_result["type"] == "symbolic":
            # Handle symbolic model format
            metrics = {
                "r2": model_result.get("r2", 0),
                "mse": model_result.get("mse", 0)
            }
            
        r2 = metrics.get("r2", 0)
        mse = metrics.get("mse", 0)
        rmse = np.sqrt(mse) if mse is not None else 0
        
        # Display metrics
        self.r2_label.setText(f"{r2:.4f}")
        self.mse_label.setText(f"{mse:.4f}")
        self.optimal_label.setText(f"{rmse:.4f}")
        
        # Display equation if available
        equation = model_result.get("equation", "")
        if equation:
            self.equation_label.setText(equation)
        else:
            self.equation_label.setText("Not available for this model type")
