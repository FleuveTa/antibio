import os
import ctypes
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit

# Path to the bin directory of your Julia installation
julia_bin_path = "D:\\Julia-1.11.5\\bin"

# Add the bin directory to PATH
os.environ["PATH"] += ";" + julia_bin_path

# Load each DLL file in the bin directory
for dll_path in glob.glob(os.path.join(julia_bin_path, "*.dll")):
    try:
        ctypes.CDLL(dll_path)
        print(f"Loaded {dll_path} successfully.")
    except OSError as e:
        print(f"Could not load {dll_path}: {e}")

# Now try to import PySR
try:
    from pysr import PySRRegressor
    SYMBOLIC_AVAILABLE = True
    print("PySR imported successfully.")
except ImportError as e:
    SYMBOLIC_AVAILABLE = False
    print(f"Could not import PySR: {e}")

class RegressionModels:
    """
    Class for fitting different regression models to parameter data
    """
    @staticmethod
    def gaussian_regression(x, y):
        """
        Fit Gaussian regression model: y = a * exp(-((x - b)^2) / (2 * c^2))

        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)

        Returns:
            Dictionary with model, parameters, metrics, and predictions
        """
        def gaussian(x, a, b, c):
            return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

        # Initial parameter guess
        p0 = [max(y), np.mean(x), np.std(x)]

        try:
            # Fit the model
            popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=10000)

            # Make predictions
            y_pred = gaussian(x, *popt)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # Optimal parameter value is at the peak of the Gaussian (parameter b)
            optimal_x = popt[1]

            return {
                "model": gaussian,
                "model_params": popt,
                "type": "gaussian",
                "equation": f"y = {popt[0]:.4f} * exp(-((x - {popt[1]:.4f})^2) / (2 * {popt[2]:.4f}^2))",
                "params": {"a": popt[0], "b": popt[1], "c": popt[2]},
                "metrics": {"r2": r2, "mse": mse},
                "predictions": y_pred,
                "optimal_x": optimal_x
            }
        except Exception as e:
            print(f"Error fitting Gaussian model: {e}")
            return None

    @staticmethod
    def sigmoid_regression(x, y):
        """
        Fit sigmoid regression model: y = a / (1 + exp(-b * (x - c))) + d

        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)

        Returns:
            Dictionary with model, parameters, metrics, and predictions
        """
        def sigmoid(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d

        # Initial parameter guess
        p0 = [max(y) - min(y), 1, np.mean(x), min(y)]

        try:
            # Fit the model
            popt, pcov = curve_fit(sigmoid, x, y, p0=p0)

            # Make predictions
            y_pred = sigmoid(x, *popt)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # For sigmoid, the optimal parameter depends on the direction
            # If b > 0, the sigmoid is increasing, so optimal is at high x
            # If b < 0, the sigmoid is decreasing, so optimal is at low x
            if popt[1] > 0:
                # For increasing sigmoid, use the inflection point + some offset
                optimal_x = popt[2] + 2/popt[1]  # c + 2/b gives ~90% of max
            else:
                # For decreasing sigmoid, use the inflection point - some offset
                optimal_x = popt[2] - 2/abs(popt[1])

            # Ensure optimal_x is within the range of x
            optimal_x = max(min(optimal_x, max(x)), min(x))

            return {
                "model": sigmoid,
                "model_params": popt,
                "type": "sigmoid",
                "equation": f"y = {popt[0]:.4f} / (1 + exp(-{popt[1]:.4f} * (x - {popt[2]:.4f}))) + {popt[3]:.4f}",
                "params": {"a": popt[0], "b": popt[1], "c": popt[2], "d": popt[3]},
                "metrics": {"r2": r2, "mse": mse},
                "predictions": y_pred,
                "optimal_x": optimal_x
            }
        except Exception as e:
            print(f"Error fitting Sigmoid model: {e}")
            return None

    @staticmethod
    def polynomial_regression_2(x, y):
        """
        Fit polynomial regression model of degree 2: y = ax^2 + bx + c

        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)

        Returns:
            Dictionary with model, parameters, metrics, and predictions
        """
        try:
            # Reshape x for sklearn
            x_reshaped = x.reshape(-1, 1)

            # Create polynomial features
            poly_features = PolynomialFeatures(degree=2, include_bias=True)

            # Create pipeline
            model = Pipeline([
                ("poly_features", poly_features),
                ("linear_regression", LinearRegression())
            ])

            # Fit the model
            model.fit(x_reshaped, y)

            # Make predictions
            y_pred = model.predict(x_reshaped)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # Get coefficients
            linear_model = model.named_steps['linear_regression']
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_

            # Create equation string
            if len(coefficients) == 3:  # Should be 3 for degree 2
                a, b, c = coefficients[2], coefficients[1], intercept
                equation = f"{a:.6f}x^2 + {b:.6f}x + {c:.6f}"
            else:
                # Fallback if coefficients don't match expected pattern
                equation = "polynomial degree 2"

            # Find optimal parameter value (maximum y value)
            # For quadratic function, the maximum/minimum is at x = -b/(2a)
            if len(coefficients) == 3 and coefficients[2] != 0:  # If a != 0
                a = coefficients[2]
                b = coefficients[1]
                critical_point = -b / (2 * a)

                # Check if it's a maximum (a < 0) or minimum (a > 0)
                if a < 0:  # It's a maximum
                    optimal_x = critical_point
                else:  # It's a minimum, so optimal is at one of the endpoints
                    y_start = model.predict(np.array([[min(x)]]))[0]
                    y_end = model.predict(np.array([[max(x)]]))[0]
                    optimal_x = min(x) if y_start > y_end else max(x)
            else:
                # Fallback: evaluate on a grid
                x_dense = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
                y_dense = model.predict(x_dense)
                optimal_idx = np.argmax(y_dense)
                optimal_x = x_dense[optimal_idx][0]

            # Ensure optimal_x is within the range of x
            optimal_x = max(min(optimal_x, max(x)), min(x))

            return {
                "model": model,
                "type": "polynomial2",
                "equation": f"y = {equation}",
                "params": {"coefficients": coefficients.tolist(), "intercept": float(intercept)},
                "metrics": {"r2": r2, "mse": mse},
                "predictions": y_pred,
                "optimal_x": optimal_x
            }
        except Exception as e:
            print(f"Error fitting Polynomial (degree 2) model: {e}")
            return None

    @staticmethod
    def polynomial_regression_3(x, y):
        """
        Fit polynomial regression model of degree 3: y = ax^3 + bx^2 + cx + d

        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)

        Returns:
            Dictionary with model, parameters, metrics, and predictions
        """
        try:
            # Reshape x for sklearn
            x_reshaped = x.reshape(-1, 1)

            # Create polynomial features
            poly_features = PolynomialFeatures(degree=3, include_bias=True)

            # Create pipeline
            model = Pipeline([
                ("poly_features", poly_features),
                ("linear_regression", LinearRegression())
            ])

            # Fit the model
            model.fit(x_reshaped, y)

            # Make predictions
            y_pred = model.predict(x_reshaped)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # Get coefficients
            linear_model = model.named_steps['linear_regression']
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_

            # Create equation string
            if len(coefficients) == 4:  # Should be 4 for degree 3
                a, b, c, d = coefficients[3], coefficients[2], coefficients[1], intercept
                equation = f"{a:.6f}x^3 + {b:.6f}x^2 + {c:.6f}x + {d:.6f}"
            else:
                # Fallback if coefficients don't match expected pattern
                equation = "polynomial degree 3"

            # Find optimal parameter value (maximum y value)
            # For cubic functions, analytical solution is complex
            # So we'll use a numerical approach
            x_dense = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
            y_dense = model.predict(x_dense)
            optimal_idx = np.argmax(y_dense)
            optimal_x = x_dense[optimal_idx][0]

            # Ensure optimal_x is within the range of x
            optimal_x = max(min(optimal_x, max(x)), min(x))

            return {
                "model": model,
                "type": "polynomial3",
                "equation": f"y = {equation}",
                "params": {"coefficients": coefficients.tolist(), "intercept": float(intercept)},
                "metrics": {"r2": r2, "mse": mse},
                "predictions": y_pred,
                "optimal_x": optimal_x
            }
        except Exception as e:
            print(f"Error fitting Polynomial (degree 3) model: {e}")
            return None

    @staticmethod
    def symbolic_regression(x, y):
        """
        Fit symbolic regression model using PySR
        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)
        Returns:
            Dictionary với model, parameters, metrics, và predictions
        """
        try:
            model = PySRRegressor(
                niterations=40,
                binary_operators=["+", "-", "*", "/", "^"],
                unary_operators=["exp", "log", "sqrt"],
                model_selection="best",
                maxsize=20,
                verbosity=0
            )
            x_reshaped = x.reshape(-1, 1)
            model.fit(x_reshaped, y)
            y_pred = model.predict(x_reshaped)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Extract just the equation string from the best model
            best_model = model.get_best()
            if hasattr(best_model, 'sympy_format'):
                equation = str(best_model.sympy_format)
            else:
                equation = str(best_model)
            
            # Tìm giá trị tối ưu (giá trị x cho y lớn nhất)
            x_dense = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
            y_dense = model.predict(x_dense)
            optimal_idx = np.argmax(y_dense)
            optimal_x = x_dense[optimal_idx][0]
            
            return {
                "model": model,
                "type": "symbolic",
                "equation": equation,
                "params": {"symbolic_expression": equation},  # Add params for consistency
                "model_params": [],  # Empty list for compatibility
                "metrics": {"r2": r2, "mse": mse},
                "predictions": y_pred,
                "optimal_x": optimal_x
            }
        except Exception as e:
            print(f"Error fitting Symbolic Regression model: {e}")
            return None

    @staticmethod
    def find_best_model(x, y):
        """
        Find the best regression model based on R² score

        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)

        Returns:
            The best model based on R² score
        """
        models = []
        errors = []

        # Fit polynomial models
        try:
            poly2_model = RegressionModels.polynomial_regression_2(x, y)
            if poly2_model:
                models.append(poly2_model)
        except Exception as e:
            errors.append(f"Polynomial (degree 2) Regression failed: {str(e)}")

        try:
            poly3_model = RegressionModels.polynomial_regression_3(x, y)
            if poly3_model:
                models.append(poly3_model)
        except Exception as e:
            errors.append(f"Polynomial (degree 3) Regression failed: {str(e)}")

        # Fit Gaussian model
        try:
            gaussian_model = RegressionModels.gaussian_regression(x, y)
            if gaussian_model:
                models.append(gaussian_model)
        except Exception as e:
            errors.append(f"Gaussian Regression failed: {str(e)}")

        # Fit Sigmoid model
        try:
            sigmoid_model = RegressionModels.sigmoid_regression(x, y)
            if sigmoid_model:
                models.append(sigmoid_model)
        except Exception as e:
            errors.append(f"Sigmoid Regression failed: {str(e)}")

        # Fit Symbolic model
        try:
            symbolic_model = RegressionModels.symbolic_regression(x, y)
            if symbolic_model:
                models.append(symbolic_model)
        except Exception as e:
            errors.append(f"Symbolic Regression failed: {str(e)}")

        # Find the best model based on R² score
        if models:
            best_model = max(models, key=lambda m: m["metrics"]["r2"])
            print(f"Best model found: {best_model['type']} with R² = {best_model['metrics']['r2']:.4f}")
            return best_model
        # If no models were successful, print all errors
        print("\nAll regression models failed:")
        for error in errors:
            print(f"- {error}")
        return None

