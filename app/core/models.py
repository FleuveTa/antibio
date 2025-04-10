import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb


class AntibioticClassifier:
    """
    Class for training and evaluating antibiotic detection models
    """
    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances = None
    
    def get_model(self):
        """
        Initialize the model based on the selected type
        """
        if self.model_type == "svm":
            return SVC(probability=True)
        elif self.model_type == "rf":
            return RandomForestClassifier()
        elif self.model_type == "gb":
            return GradientBoostingClassifier()
        elif self.model_type == "xgb":
            return xgb.XGBClassifier()
        elif self.model_type == "lgb":
            return lgb.LGBMClassifier()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model and evaluate its performance
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train the model
        self.model = self.get_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importances if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = self.model.feature_importances_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        return {
            "model": self.model,
            "metrics": metrics,
            "feature_importances": self.feature_importances,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
    
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
            # Add grids for other model types
        
        # Initialize the model
        base_model = self.get_model()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_scaled, y)
        
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