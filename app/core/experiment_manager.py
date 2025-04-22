import os
import sqlite3
import pandas as pd
from datetime import datetime
import json
import shutil
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter
from scipy.stats import zscore

class ExperimentManager:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
        self._setup_database()

    def _setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "raw",           # Raw data
            "processed",     # Processed data
            "features",      # Extracted features
            "models",        # Trained models
            "transformers"   # Preprocessing transformers
        ]

        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_database(self):
        """Initialize SQLite database"""
        self.db_path = str(self.base_dir / "experiments.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create experiments table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            label_column TEXT,
            parameters TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Check if we need to migrate from old schema
        self._migrate_database()

        # Create preprocessing_steps table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS preprocessing_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            step_name TEXT NOT NULL,
            parameters TEXT,
            transformer_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')

        # Create features table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            feature_name TEXT NOT NULL,
            value REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')

        # Create labels table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            sample_id TEXT NOT NULL,
            label_value REAL,
            label_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')

        self.conn.commit()

    def _migrate_database(self):
        """Check if we need to migrate from old schema to new schema"""
        try:
            # Check experiments table columns
            self.cursor.execute("PRAGMA table_info(experiments)")
            exp_columns = [column[1] for column in self.cursor.fetchall()]

            # Check if we need to add label_column
            if 'label_column' not in exp_columns:
                print("Adding label_column to experiments table...")
                self.cursor.execute("ALTER TABLE experiments ADD COLUMN label_column TEXT")
                self.conn.commit()
                print("Added label_column to experiments table.")

            # Check if we need to add status column
            if 'status' not in exp_columns:
                print("Adding status column to experiments table...")
                self.cursor.execute("ALTER TABLE experiments ADD COLUMN status TEXT DEFAULT 'active'")
                self.conn.commit()
                print("Added status column to experiments table.")

            # If we have both old and new columns, migrate data
            if 'label_file_path' in exp_columns and 'label_column' in exp_columns:
                print("Migrating data from label_file_path to label_column...")
                self.cursor.execute("UPDATE experiments SET label_column = NULL WHERE label_column IS NULL")
                self.conn.commit()
                print("Data migration completed.")

            # Check preprocessing_steps table columns
            # First check if the table exists
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='preprocessing_steps'")
            if self.cursor.fetchone():
                self.cursor.execute("PRAGMA table_info(preprocessing_steps)")
                preproc_columns = [column[1] for column in self.cursor.fetchall()]

                # Check if transformer_path column exists
                if 'transformer_path' not in preproc_columns:
                    print("Adding transformer_path column to preprocessing_steps table...")
                    self.cursor.execute("ALTER TABLE preprocessing_steps ADD COLUMN transformer_path TEXT")
                    self.conn.commit()
                    print("Added transformer_path column to preprocessing_steps table.")

        except Exception as e:
            print(f"Error during database migration: {e}")

    def create_experiment(self, name, file_path, label_column=None, description="", parameters=None):
        """Create a new experiment"""
        # Create experiment directory
        exp_dir = self.base_dir / "raw" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Copy data file to experiment directory
        new_file_path = exp_dir / os.path.basename(file_path)
        shutil.copy2(file_path, new_file_path)

        # Save to database
        self.cursor.execute('''
        INSERT INTO experiments (name, date, description, file_path, label_column, parameters, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              description, str(new_file_path),
              label_column,
              json.dumps(parameters) if parameters else None,
              'active'))

        self.conn.commit()
        return self.cursor.lastrowid

    def add_preprocessing_step(self, experiment_id, step_name, parameters=None, transformer=None):
        """Add a preprocessing step to the experiment"""
        # Save transformer if provided
        transformer_path = None
        if transformer is not None:
            transformers_dir = self.base_dir / "transformers" / f"experiment_{experiment_id}"
            transformers_dir.mkdir(parents=True, exist_ok=True)
            transformer_path = transformers_dir / f"{step_name}.pkl"
            with open(transformer_path, 'wb') as f:
                pickle.dump(transformer, f)

        # Save to database
        self.cursor.execute('''
        INSERT INTO preprocessing_steps (experiment_id, step_name, parameters, transformer_path)
        VALUES (?, ?, ?, ?)
        ''', (experiment_id, step_name,
              json.dumps(parameters) if parameters else None,
              str(transformer_path) if transformer_path else None))

        self.conn.commit()
        return self.cursor.lastrowid

    def get_preprocessing_transformers(self, experiment_id):
        """Get all preprocessing transformers for an experiment"""
        self.cursor.execute('''
        SELECT step_name, transformer_path
        FROM preprocessing_steps
        WHERE experiment_id = ? AND transformer_path IS NOT NULL
        ''', (experiment_id,))

        transformers = {}
        for step_name, transformer_path in self.cursor.fetchall():
            if transformer_path and os.path.exists(transformer_path):
                with open(transformer_path, 'rb') as f:
                    transformers[step_name] = pickle.load(f)

        return transformers

    def apply_preprocessing_transformers(self, experiment_id, data):
        """Apply saved preprocessing transformers to new data"""
        transformers = self.get_preprocessing_transformers(experiment_id)

        for step_name, transformer in transformers.items():
            if step_name == "normalize":
                data = transformer.transform(data)
            elif step_name == "impute":
                data = transformer.transform(data)
            # Add other transformer types as needed

        return data

    def save_processed_data(self, experiment_id, df, step_name):
        """Save processed data"""
        # Create processed directory if it doesn't exist
        processed_dir = self.base_dir / "processed" / f"experiment_{experiment_id}"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        file_path = processed_dir / f"{step_name}.csv"
        df.to_csv(file_path, index=False)

        return str(file_path)

    def get_processed_data(self, experiment_id, step_name="preprocessed"):
        """Load processed data for an experiment"""
        processed_dir = self.base_dir / "processed" / f"experiment_{experiment_id}"
        file_path = processed_dir / f"{step_name}.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded processed data from {file_path}")
                return df
            except Exception as e:
                print(f"Error loading processed data: {e}")
                return None
        else:
            print(f"No processed data found at {file_path}")
            return None

    def save_features(self, experiment_id, features):
        """Save extracted features"""
        # Create features directory if it doesn't exist
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        file_path = features_dir / "features.csv"
        pd.DataFrame(features).to_csv(file_path, index=False)

        return str(file_path)

    def get_experiment(self, experiment_id):
        """Get experiment information"""
        try:
            # Debug information
            print(f"Getting experiment with ID: {experiment_id}")

            self.cursor.execute('''
            SELECT * FROM experiments WHERE id = ?
            ''', (experiment_id,))

            experiment = self.cursor.fetchone()
            if experiment:
                # Check if we're using the new schema or old schema
                columns = [column[0] for column in self.cursor.description]
                print(f"Database columns: {columns}")
                print(f"Experiment data: {experiment}")

                # Create result dictionary safely
                result = {'id': experiment[0]}

                # Add basic fields with error handling
                try:
                    result['name'] = experiment[1] if experiment[1] is not None else "Unnamed Experiment"
                except (IndexError, TypeError) as e:
                    print(f"Error getting name: {e}")
                    result['name'] = "Unnamed Experiment"

                try:
                    result['date'] = experiment[2] if experiment[2] is not None else "Unknown Date"
                except (IndexError, TypeError) as e:
                    print(f"Error getting date: {e}")
                    result['date'] = "Unknown Date"

                try:
                    result['description'] = experiment[3] if experiment[3] is not None else ""
                except (IndexError, TypeError) as e:
                    print(f"Error getting description: {e}")
                    result['description'] = ""

                try:
                    result['file_path'] = experiment[4] if experiment[4] is not None else ""
                except (IndexError, TypeError) as e:
                    print(f"Error getting file_path: {e}")
                    result['file_path'] = ""

                # Add remaining fields based on schema
                if 'label_column' in columns:
                    try:
                        label_column_index = columns.index('label_column')
                        result['label_column'] = experiment[label_column_index]
                    except (IndexError, TypeError) as e:
                        print(f"Error getting label_column: {e}")
                        result['label_column'] = None
                elif 'label_file_path' in columns:
                    try:
                        label_file_path_index = columns.index('label_file_path')
                        result['label_file_path'] = experiment[label_file_path_index]
                    except (IndexError, TypeError) as e:
                        print(f"Error getting label_file_path: {e}")
                        result['label_file_path'] = None

                # Add parameters
                try:
                    parameters_index = columns.index('parameters')
                    if experiment[parameters_index] and isinstance(experiment[parameters_index], str):
                        try:
                            result['parameters'] = json.loads(experiment[parameters_index])
                        except json.JSONDecodeError as e:
                            print(f"Error parsing parameters JSON: {e}")
                            result['parameters'] = None
                    else:
                        result['parameters'] = None
                except (IndexError, ValueError) as e:
                    print(f"Error getting parameters: {e}")
                    result['parameters'] = None

                # Add status if available
                if 'status' in columns:
                    try:
                        status_index = columns.index('status')
                        result['status'] = experiment[status_index]
                    except (IndexError, TypeError) as e:
                        print(f"Error getting status: {e}")
                        result['status'] = 'active'  # Default status

                # Add created_at if available
                if 'created_at' in columns:
                    try:
                        created_at_index = columns.index('created_at')
                        result['created_at'] = experiment[created_at_index] if experiment[created_at_index] else result['date']
                    except (IndexError, TypeError) as e:
                        print(f"Error getting created_at: {e}")
                        result['created_at'] = result['date']  # Use date as fallback

                print(f"Returning experiment data: {result}")
                return result
            else:
                print(f"No experiment found with ID: {experiment_id}")
                return None
        except Exception as e:
            print(f"Error in get_experiment: {e}")
            import traceback
            traceback.print_exc()
            return None

    def list_experiments(self, active_only=False):  # Changed default to False to show all experiments
        """List all experiments"""
        try:
            # Debug: Print database path
            print(f"Database path: {self.db_path}")

            # Check if status column exists
            self.cursor.execute("PRAGMA table_info(experiments)")
            columns = [column[1] for column in self.cursor.fetchall()]
            has_status_column = 'status' in columns

            print(f"Database columns: {columns}")
            print(f"Has status column: {has_status_column}")

            # CRITICAL FIX: Use a simpler query that works regardless of schema
            self.cursor.execute('SELECT * FROM experiments ORDER BY date DESC')
            rows = self.cursor.fetchall()
            print(f"Found {len(rows)} experiments in database")

            # Get column names for mapping
            column_names = [column[0] for column in self.cursor.description]
            print(f"Column names: {column_names}")

            experiments = []
            for row in rows:
                # Create a dictionary mapping column names to values
                exp_dict = {column_names[i]: row[i] for i in range(len(column_names))}

                # Create a standardized experiment dictionary
                exp = {
                    'id': exp_dict['id'],
                    'name': exp_dict['name'],
                    'date': exp_dict['date'],
                    'description': exp_dict.get('description', ''),
                }

                # Add status if available
                if 'status' in exp_dict:
                    exp['status'] = exp_dict['status']
                else:
                    exp['status'] = 'active'  # Default status

                # Only filter by status if active_only is True and we have status
                if active_only and exp.get('status') != 'active':
                    continue

                experiments.append(exp)
                print(f"Added experiment: {exp['name']} (ID: {exp['id']})")

            return experiments
        except Exception as e:
            print(f"Error in list_experiments: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def save_model(self, experiment_id, model, model_type, metrics=None):
        """Save a trained model"""
        # Create models directory if it doesn't exist
        models_dir = self.base_dir / "models" / f"experiment_{experiment_id}"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / f"{model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metrics if provided
        if metrics:
            metrics_path = models_dir / f"{model_type}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

        # Update experiment status
        self.cursor.execute('''
        UPDATE experiments SET status = 'completed' WHERE id = ?
        ''', (experiment_id,))
        self.conn.commit()

        return str(model_path)

    def load_model(self, experiment_id, model_type):
        """Load a trained model"""
        model_path = self.base_dir / "models" / f"experiment_{experiment_id}" / f"{model_type}_model.pkl"
        metrics_path = self.base_dir / "models" / f"experiment_{experiment_id}" / f"{model_type}_metrics.json"

        if not model_path.exists():
            return None, None

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load metrics if available
        metrics = None
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        return model, metrics

    def get_preprocessing_steps(self, experiment_id):
        """Get preprocessing steps for an experiment"""
        self.cursor.execute('''
        SELECT * FROM preprocessing_steps WHERE experiment_id = ?
        ''', (experiment_id,))

        steps = []
        for step in self.cursor.fetchall():
            steps.append({
                'id': step[0],
                'experiment_id': step[1],
                'step_name': step[2],
                'parameters': json.loads(step[3]) if step[3] else None,
                'transformer_path': step[4],
                'created_at': step[5]
            })
        return steps

    # The features and labels tables have been removed from the database schema
    # Features and labels are now stored in CSV files instead

    def save_features(self, experiment_id, features, metadata=None):
        """Save features for an experiment to CSV file

        Args:
            experiment_id: ID of the experiment
            features: Dictionary of features or DataFrame of features
            metadata: Optional dictionary of metadata about the features

        Returns:
            Path to the saved features file
        """
        # Create features directory if it doesn't exist
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Convert features to DataFrame if it's a dictionary
        if isinstance(features, dict):
            # Convert flat dictionary to DataFrame with one row
            df = pd.DataFrame([features])
        else:
            df = features

        # Save features to CSV
        file_path = features_dir / "features.csv"
        df.to_csv(file_path, index=False)

        # Save metadata if provided
        if metadata:
            metadata_path = features_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Update experiment status
        self.cursor.execute(
            "UPDATE experiments SET status = ? WHERE id = ?",
            ("features_extracted", experiment_id)
        )
        self.conn.commit()

        print(f"Features saved to {file_path}")
        return str(file_path)

    def save_feature_matrix(self, experiment_id, feature_matrix, metadata=None):
        """Save a feature matrix for an experiment to CSV file

        This is specifically for the sample-by-sample feature extraction where
        each row represents a sample with its features and metadata.

        Args:
            experiment_id: ID of the experiment
            feature_matrix: DataFrame where each row is a sample with features and metadata
            metadata: Optional dictionary of metadata about the feature matrix

        Returns:
            Path to the saved feature matrix file
        """
        # Create features directory if it doesn't exist
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Save feature matrix to CSV
        file_path = features_dir / "feature_matrix.csv"
        feature_matrix.to_csv(file_path, index=False)

        # Save metadata if provided
        if metadata:
            metadata_path = features_dir / "matrix_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Update experiment status
        self.cursor.execute(
            "UPDATE experiments SET status = ? WHERE id = ?",
            ("features_extracted", experiment_id)
        )
        self.conn.commit()

        print(f"Feature matrix saved to {file_path}")
        return str(file_path)

    def get_features(self, experiment_id):
        """Get features for an experiment from CSV file

        Args:
            experiment_id: ID of the experiment

        Returns:
            DataFrame of features or empty list if not found
        """
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
        file_path = features_dir / "features.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                print(f"Error loading features: {e}")

        return pd.DataFrame()

    def get_feature_matrix(self, experiment_id):
        """Get feature matrix for an experiment from CSV file

        This is for the sample-by-sample feature extraction format where
        each row represents a sample with its features and metadata.

        Args:
            experiment_id: ID of the experiment

        Returns:
            DataFrame of feature matrix or empty DataFrame if not found
        """
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
        file_path = features_dir / "feature_matrix.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                print(f"Error loading feature matrix: {e}")

        # If feature matrix doesn't exist, try to load regular features
        # and check if they have metadata columns
        regular_features = self.get_features(experiment_id)
        if not regular_features.empty and any(col in regular_features.columns for col in ['concentration', 'antibiotic']):
            return regular_features

        return pd.DataFrame()

    def get_feature_metadata(self, experiment_id, matrix=False):
        """Get metadata about features for an experiment

        Args:
            experiment_id: ID of the experiment
            matrix: Whether to get metadata for the feature matrix (True) or regular features (False)

        Returns:
            Dictionary of metadata or empty dict if not found
        """
        features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"

        # Determine which metadata file to load
        if matrix:
            metadata_path = features_dir / "matrix_metadata.json"
        else:
            metadata_path = features_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feature metadata: {e}")

        # If matrix metadata doesn't exist but was requested, try regular metadata
        if matrix and not metadata_path.exists():
            regular_metadata_path = features_dir / "metadata.json"
            if regular_metadata_path.exists():
                try:
                    with open(regular_metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # Check if this might be matrix metadata
                        if 'sample_count' in metadata or 'metadata_columns' in metadata:
                            return metadata
                except Exception:
                    pass

        return {}

    def add_labels(self, experiment_id, labels_df):
        """Add labels to the experiment by saving to CSV"""
        # Create labels directory if it doesn't exist
        labels_dir = self.base_dir / "processed" / f"experiment_{experiment_id}"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Save labels to CSV
        file_path = labels_dir / "labels.csv"
        labels_df.to_csv(file_path, index=False)

        return str(file_path)

    def get_labels(self, experiment_id):
        """Get labels for an experiment from CSV file"""
        labels_dir = self.base_dir / "processed" / f"experiment_{experiment_id}"
        file_path = labels_dir / "labels.csv"

        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                return df.to_dict('records')
            except Exception as e:
                print(f"Error loading labels: {e}")

        return []

    def delete_experiment(self, experiment_id):
        """Delete an experiment and all associated data"""
        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")

            # Delete associated data first (due to foreign key constraints)
            self.cursor.execute("DELETE FROM preprocessing_steps WHERE experiment_id = ?", (experiment_id,))

            # Get experiment details for file cleanup
            self.cursor.execute("SELECT file_path FROM experiments WHERE id = ?", (experiment_id,))
            experiment = self.cursor.fetchone()

            # Delete the experiment
            self.cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

            # Commit transaction
            self.conn.commit()

            # Clean up associated files
            if experiment:
                # Get the file path from the experiment
                file_path = experiment[0]
                if file_path and os.path.exists(file_path):
                    # Get the raw directory containing the experiment data
                    raw_dir = os.path.dirname(file_path)
                    if os.path.exists(raw_dir) and raw_dir.startswith(str(self.base_dir / "raw")):
                        print(f"Deleting raw data directory: {raw_dir}")
                        import shutil
                        shutil.rmtree(raw_dir)
                    else:
                        # If we can't delete the whole directory, at least delete the file
                        print(f"Deleting raw data file: {file_path}")
                        os.remove(file_path)

                # Delete processed data directory
                processed_dir = self.base_dir / "processed" / f"experiment_{experiment_id}"
                if processed_dir.exists():
                    print(f"Deleting processed data directory: {processed_dir}")
                    import shutil
                    shutil.rmtree(processed_dir)

                # Delete features directory
                features_dir = self.base_dir / "features" / f"experiment_{experiment_id}"
                if features_dir.exists():
                    print(f"Deleting features directory: {features_dir}")
                    import shutil
                    shutil.rmtree(features_dir)

                # Delete transformers directory
                transformers_dir = self.base_dir / "transformers" / f"experiment_{experiment_id}"
                if transformers_dir.exists():
                    print(f"Deleting transformers directory: {transformers_dir}")
                    import shutil
                    shutil.rmtree(transformers_dir)

                # Delete models directory
                models_dir = self.base_dir / "models" / f"experiment_{experiment_id}"
                if models_dir.exists():
                    print(f"Deleting models directory: {models_dir}")
                    import shutil
                    shutil.rmtree(models_dir)

            return True
        except Exception as e:
            # Rollback in case of error
            self.conn.rollback()
            print(f"Error deleting experiment: {e}")
            return False

    def close(self):
        """Close database connection"""
        self.conn.close()