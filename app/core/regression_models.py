import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit


class RegressionModels:
    """
    Class for fitting different regression models to parameter data
    """
    @staticmethod
    def linear_regression(x, y):
        """
        Fit linear regression model: y = mx + b
        
        Args:
            x: Independent variable (parameter values)
            y: Dependent variable (peak current values)
            
        Returns:
            Dictionary with model, parameters, metrics, and predictions
        """
        model = LinearRegression()
        x_reshaped = x.reshape(-1, 1)
        model.fit(x_reshaped, y)
        
        # Make predictions
        y_pred = model.predict(x_reshaped)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        # Get model parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Find optimal parameter value (maximum y value)
        if slope > 0:
            # If slope is positive, maximum is at highest x
            optimal_x = max(x)
        elif slope < 0:
            # If slope is negative, maximum is at lowest x
            optimal_x = min(x)
        else:
            # If slope is zero, any x is optimal
            optimal_x = np.mean(x)
        
        return {
            "model": model,
            "type": "linear",
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "params": {"slope": slope, "intercept": intercept},
            "metrics": {"r2": r2, "mse": mse},
            "predictions": y_pred,
            "optimal_x": optimal_x
        }
    
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
            popt, pcov = curve_fit(gaussian, x, y, p0=p0)
            
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
        
        # Fit linear model
        linear_model = RegressionModels.linear_regression(x, y)
        if linear_model:
            models.append(linear_model)
        
        # Fit Gaussian model
        gaussian_model = RegressionModels.gaussian_regression(x, y)
        if gaussian_model:
            models.append(gaussian_model)
        
        # Fit Sigmoid model
        sigmoid_model = RegressionModels.sigmoid_regression(x, y)
        if sigmoid_model:
            models.append(sigmoid_model)
        
        # Find the best model based on R² score
        if models:
            best_model = max(models, key=lambda m: m["metrics"]["r2"])
            return best_model
        
        return None
