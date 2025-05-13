import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
# Temporarily disable scaling
# from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, explained_variance_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


class AntibioticClassifier:
    """
    Class for training and evaluating antibiotic detection models
    """
    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.model = None
        # Temporarily disable scaling
        # self.scaler = StandardScaler()
        self.scaler = None
        self.feature_importances = None

    def get_model(self):
        """
        Initialize the model based on the selected type
        """
        if self.model_type == "svm":
            # Improved SVM parameters for better performance with electrochemical data
            return SVC(probability=True, C=10.0, gamma='auto', class_weight='balanced',
                       kernel='rbf', decision_function_shape='ovr')
        elif self.model_type == "rf":
            return RandomForestClassifier()
        elif self.model_type == "gb":
            return GradientBoostingClassifier()
        elif self.model_type == "xgb":
            return xgb.XGBClassifier()
        elif self.model_type == "lgb":
            # Add specific parameters to avoid warnings and improve stability
            return lgb.LGBMClassifier(min_child_samples=5, min_data_in_leaf=5, min_child_weight=1e-3,
                                      verbose=-1, force_col_wise=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model and evaluate its performance
        """
        # Print data information for debugging
        print(f"Training classification model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Convert all elements to string to avoid mixed type comparison
        try:
            y_str = np.array([str(val) for val in y])
            print(f"Target classes: {np.unique(y_str)}")
        except Exception as e:
            print(f"Error displaying target classes: {e}")
        
        # For XGBoost, we need to encode string labels to integers
        if self.model_type == "xgb":
            print("XGBoost requires numeric class labels. Encoding string labels...")
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            # Store the encoder for later use
            self.label_encoder = label_encoder
            print(f"Encoded classes: {np.unique(y_encoded)} for original classes: {label_encoder.classes_}")
            y = y_encoded

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Skip scaling for now
        if self.scaler is not None:
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Use original data without scaling
            X_train_scaled = X_train
            X_test_scaled = X_test
            print("Scaling disabled - using original data without normalization")

        # Store feature names if available (for LightGBM compatibility)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            print(f"Stored {len(self.feature_names)} feature names for model training")
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            print(f"Created {len(self.feature_names)} generic feature names")

        # Initialize and train the model
        self.model = self.get_model()

        # For LightGBM, we need to pass feature names
        if self.model_type == "lgb":
            print("Training LightGBM model with feature names")
            # Convert to DataFrame with feature names
            X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            self.model.fit(X_train_df, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)

        # Get feature importances if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = self.model.feature_importances_

        # Evaluate the model
        try:
            if self.model_type == "lgb":
                # For LightGBM, we need to use feature names
                X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)
                y_pred = self.model.predict(X_test_df)
                # Get prediction probabilities
                y_pred_proba = self.model.predict_proba(X_test_df)
            else:
                y_pred = self.model.predict(X_test_scaled)
                # Get prediction probabilities
                y_pred_proba = self.model.predict_proba(X_test_scaled)

            # For XGBoost, we need to decode the predictions back to original labels
            if self.model_type == "xgb" and hasattr(self, 'label_encoder'):
                # Decode the predicted labels
                y_pred = self.label_encoder.inverse_transform(y_pred.astype(int))
                # For test set, we need to decode the original labels for comparison
                y_test_original = self.label_encoder.inverse_transform(y_test.astype(int))
                y_test = y_test_original
        except Exception as e:
            print(f"Error getting prediction probabilities: {e}")
            # Find number of classes safely, avoiding mixed type comparison
            try:
                # Convert y_test to string to get a safe count of unique values
                y_test_str = np.array([str(val) for val in y_test])
                num_classes = len(np.unique(y_test_str))
                y_pred_proba = np.zeros((len(y_test), num_classes))
            except Exception as e2:
                print(f"Error creating dummy probabilities: {e2}")
                # Fallback: just create a dummy binary array
                y_pred_proba = np.zeros((len(y_test), 2))

        # Calculate simple metrics
        try:
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred)
            }

            # Calculate additional metrics
            # For binary classification
            try:
                # Convert y_test to string array to avoid mixed type comparisons
                y_test_str = np.array([str(val) for val in y_test])
                num_classes = len(np.unique(y_test_str))
                
                if num_classes == 2:
                    # Precision, recall, f1
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    metrics["precision"] = precision_score(y_test, y_pred, average='binary')
                    metrics["recall"] = recall_score(y_test, y_pred, average='binary')
                    metrics["f1"] = f1_score(y_test, y_pred, average='binary')

                    # ROC AUC - only for binary classification
                    try:
                        # Get probability of positive class
                        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                            # For models that return probabilities for all classes
                            pos_proba = y_pred_proba[:, 1]
                        else:
                            # For models that return probability of positive class directly
                            pos_proba = y_pred_proba

                        metrics["roc_auc"] = roc_auc_score(y_test, pos_proba)
                    except Exception as e:
                        print(f"Could not calculate ROC AUC: {e}")
                else:
                    # For multiclass classification
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
                    metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
                    metrics["f1"] = f1_score(y_test, y_pred, average='weighted')
            except Exception as e:
                print(f"Error determining number of classes: {e}")
                # Default to multiclass metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
                metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
                metrics["f1"] = f1_score(y_test, y_pred, average='weighted')
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()  # Convert to list for JSON serialization
        except Exception as e:
            print(f"Error calculating confusion matrix: {e}")
            metrics["confusion_matrix"] = [[0]]

        result = {
            "model": self.model,
            "metrics": metrics,
            "feature_importances": self.feature_importances,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "model_type": self.model_type  # Add model type for reference
        }

        # Add model parameters for reference
        if hasattr(self.model, 'get_params'):
            result["model_params"] = self.model.get_params()

        # Add label encoder for XGBoost models
        if self.model_type == "xgb" and hasattr(self, 'label_encoder'):
            result["label_encoder"] = self.label_encoder

        return result

    def optimize_hyperparameters(self, X, y, param_grid=None):
        """
        Perform hyperparameter optimization using GridSearchCV
        """
        # Default parameter grids for different models
        if param_grid is None:
            if self.model_type == "svm":
                param_grid = {
                    'C': [1, 10, 100, 1000],  # Higher C values for better accuracy
                    'gamma': ['scale', 'auto', 0.1, 0.01],  # Include scale and auto
                    'kernel': ['rbf', 'poly', 'sigmoid'],  # Try different kernels
                    'class_weight': ['balanced', None]  # Try with and without class balancing
                }
            elif self.model_type == "rf":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == "gb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == "xgb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0.5, 1.0, 1.5]
                }
            elif self.model_type == "lgb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],  # Increased learning rates
                    'num_leaves': [15, 31, 63],  # Instead of max_depth
                    'min_child_samples': [5, 10, 20],  # Minimum samples in leaf
                    'min_child_weight': [1e-3, 1e-2, 1e-1],  # Minimum sum of instance weight
                    'subsample': [0.8, 1.0],  # Subsample ratio of training instances
                    'colsample_bytree': [0.8, 1.0]  # Subsample ratio of columns
                }
            else:
                # Default grid for unknown model types
                param_grid = {}

        # Initialize the model
        base_model = self.get_model()

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Perform grid search
        try:
            # For classification, use accuracy as the scoring metric
            scoring = 'accuracy'

            # Create and fit the grid search
            print(f"Starting grid search with parameters: {param_grid}")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_scaled, y)

            print(f"Grid search complete. Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"Error during grid search: {e}")
            # Return a default result if grid search fails
            return {
                "best_model": self.model,
                "best_params": {},
                "best_score": 0.0,
                "cv_results": {},
                "error": str(e)
            }

        # Get the best model and parameters
        self.model = grid_search.best_estimator_

        return {
            "best_model": self.model,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

    def predict(self, X):
        """
        Make predictions with the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale the features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

        return {
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }


class RegressionClassifier:
    """
    Class for training and evaluating regression models for concentration prediction
    """
    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.model = None
        # Temporarily disable scaling
        # Use RobustScaler instead of StandardScaler for regression
        # RobustScaler is less affected by outliers
        # self.scaler = RobustScaler()
        self.scaler = None
        self.feature_importances = None

    def get_model(self):
        """
        Initialize the model based on the selected type
        """
        if self.model_type == "svm":
            # Improved SVR parameters for better performance with electrochemical data
            return SVR(C=10.0, gamma='auto', epsilon=0.1, kernel='rbf')
        elif self.model_type == "rf":
            return RandomForestRegressor()
        elif self.model_type == "gb":
            return GradientBoostingRegressor()
        elif self.model_type == "xgb":
            return xgb.XGBRegressor()
        elif self.model_type == "lgb":
            # Add specific parameters to avoid warnings and improve stability
            return lgb.LGBMRegressor(min_child_samples=5, min_data_in_leaf=5, min_child_weight=1e-3,
                                     verbose=-1, force_col_wise=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model and evaluate its performance
        """
        # Print data information for debugging
        print(f"Training regression model with {X.shape[0]} samples and {X.shape[1]} features")

        # Convert y to numeric if it's not already
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y, dtype=float)
            except (ValueError, TypeError) as e:
                print(f"Error converting target to numpy array: {e}")
                # Try pandas conversion which handles more formats
                if hasattr(y, 'values'):
                    # It's a pandas Series or DataFrame
                    y = pd.to_numeric(y, errors='coerce').values
                else:
                    # Try to convert list or other iterable
                    y = pd.to_numeric(pd.Series(y), errors='coerce').values

        # Check for NaN or infinite values in X
        if np.isnan(X).any() or np.isinf(X).any():
            print("Warning: Input features contain NaN or infinite values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for NaN or infinite values in y
        if np.isnan(y).any() or np.isinf(y).any():
            print("Warning: Target variable contains NaN or infinite values. Replacing with mean.")
            # Calculate mean safely
            valid_y = y[~np.isnan(y) & ~np.isinf(y)]
            if len(valid_y) > 0:
                y_mean = np.mean(valid_y)
            else:
                y_mean = 0.0
            y = np.nan_to_num(y, nan=y_mean, posinf=y_mean, neginf=y_mean)

        # Print target variable statistics after cleaning
        print(f"Target variable range: {np.min(y)} to {np.max(y)}, mean: {np.mean(y)}, std: {np.std(y)}")

        # Check if we have enough variation in the target variable
        unique_values = np.unique(y)
        if len(unique_values) < 3:
            print(f"Warning: Target variable has only {len(unique_values)} unique values. Regression models work best with more variation.")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        # Skip scaling for now
        if self.scaler is not None:
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Use original data without scaling
            X_train_scaled = X_train
            X_test_scaled = X_test
            print("Scaling disabled - using original data without normalization")

        # Store feature names if available (for LightGBM compatibility)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            print(f"Stored {len(self.feature_names)} feature names for model training")
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            print(f"Created {len(self.feature_names)} generic feature names")

        # Initialize and train the model
        self.model = self.get_model()

        # Set model parameters based on model type
        try:
            if self.model_type == "rf":
                # Random Forest specific settings
                self.model.set_params(min_samples_leaf=2, n_estimators=100, max_depth=None)
                print("Using Random Forest with min_samples_leaf=2, n_estimators=100, max_depth=None")
            elif self.model_type == "gb":
                # Gradient Boosting specific settings
                self.model.set_params(n_estimators=100, learning_rate=0.05, max_depth=3)
                print("Using Gradient Boosting with n_estimators=100, learning_rate=0.05, max_depth=3")
            elif self.model_type == "xgb":
                # XGBoost specific settings
                self.model.set_params(n_estimators=100, learning_rate=0.05, max_depth=3, reg_alpha=0.1, reg_lambda=1.0)
                print("Using XGBoost with n_estimators=100, learning_rate=0.05, max_depth=3, reg_alpha=0.1, reg_lambda=1.0")
            elif self.model_type == "lgb":
                # LightGBM specific settings
                self.model.set_params(n_estimators=100, learning_rate=0.05, max_depth=3, reg_alpha=0.1, reg_lambda=1.0)
                print("Using LightGBM with n_estimators=100, learning_rate=0.05, max_depth=3, reg_alpha=0.1, reg_lambda=1.0")
            elif self.model_type == "svm":
                # SVM specific settings
                self.model.set_params(C=10.0, epsilon=0.1, kernel='rbf')
                print("Using SVM with C=10.0, epsilon=0.1, kernel='rbf'")
        except Exception as e:
            print(f"Error setting model parameters: {e}")
            print("Using default model parameters")

        # Fit the model
        print("Fitting regression model...")
        if self.model_type == "lgb":
            # For LightGBM, we need to pass feature names
            print("Training LightGBM regression model with feature names")
            # Convert to DataFrame with feature names
            X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            self.model.fit(X_train_df, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)
        print("Model fitting complete")

        # Get feature importances if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = self.model.feature_importances_
            # Print top 5 most important features
            if len(self.feature_importances) > 0:
                indices = np.argsort(self.feature_importances)[::-1]
                print("Top 5 feature importances:")
                for i in range(min(5, len(indices))):
                    print(f"Feature {indices[i]}: {self.feature_importances[indices[i]]:.4f}")

        # Evaluate the model
        print("Evaluating model...")
        if self.model_type == "lgb":
            # For LightGBM, we need to use feature names
            X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)
            y_pred = self.model.predict(X_test_df)
        else:
            y_pred = self.model.predict(X_test_scaled)

        # Print prediction statistics
        print(f"Predictions range: {np.min(y_pred)} to {np.max(y_pred)}, mean: {np.mean(y_pred)}, std: {np.std(y_pred)}")

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"Regression metrics - R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Calculate additional metrics
        metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "explained_variance": explained_variance_score(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred)
        }

        result = {
            "model": self.model,
            "metrics": metrics,
            "feature_importances": self.feature_importances,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "model_type": self.model_type  # Add model type for reference
        }

        # Add model parameters for reference
        if hasattr(self.model, 'get_params'):
            result["model_params"] = self.model.get_params()

        return result

    def optimize_hyperparameters(self, X, y, param_grid=None):
        """
        Perform hyperparameter optimization using GridSearchCV
        """
        # Default parameter grids for different models
        if param_grid is None:
            if self.model_type == "svm":
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'linear']
                }
            elif self.model_type == "rf":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == "gb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == "xgb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0.5, 1.0, 1.5]
                }
            elif self.model_type == "lgb":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],  # Increased learning rates
                    'num_leaves': [15, 31, 63],  # Instead of max_depth
                    'min_child_samples': [5, 10, 20],  # Minimum samples in leaf
                    'min_child_weight': [1e-3, 1e-2, 1e-1],  # Minimum sum of instance weight
                    'subsample': [0.8, 1.0],  # Subsample ratio of training instances
                    'colsample_bytree': [0.8, 1.0]  # Subsample ratio of columns
                }
            else:
                # Default grid for unknown model types
                param_grid = {}

        # Initialize the model
        base_model = self.get_model()

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Perform grid search
        try:
            # For regression, use neg_mean_squared_error as the scoring metric
            scoring = 'neg_mean_squared_error'

            # Create and fit the grid search
            print(f"Starting grid search with parameters: {param_grid}")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_scaled, y)

            print(f"Grid search complete. Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"Error during grid search: {e}")
            # Return a default result if grid search fails
            return {
                "best_model": self.model,
                "best_params": {},
                "best_score": 0.0,
                "cv_results": {},
                "error": str(e)
            }

        # Get the best model and parameters
        self.model = grid_search.best_estimator_

        return {
            "best_model": self.model,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

    def predict(self, X):
        """
        Make predictions with the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale the features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        y_pred = self.model.predict(X_scaled)

        return {
            "predictions": y_pred
        }