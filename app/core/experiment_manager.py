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
        db_path = self.base_dir / "experiments.db"
        self.conn = sqlite3.connect(db_path)
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
            # Check if label_file_path column exists
            self.cursor.execute("PRAGMA table_info(experiments)")
            columns = [column[1] for column in self.cursor.fetchall()]

            # Check if we need to add label_column
            if 'label_column' not in columns:
                print("Adding label_column to database schema...")
                self.cursor.execute("ALTER TABLE experiments ADD COLUMN label_column TEXT")
                self.conn.commit()
                print("Added label_column to database schema.")

            # Check if we need to add status column
            if 'status' not in columns:
                print("Adding status column to database schema...")
                self.cursor.execute("ALTER TABLE experiments ADD COLUMN status TEXT DEFAULT 'active'")
                self.conn.commit()
                print("Added status column to database schema.")

            # If we have both old and new columns, migrate data
            if 'label_file_path' in columns and 'label_column' in columns:
                print("Migrating data from label_file_path to label_column...")
                self.cursor.execute("UPDATE experiments SET label_column = NULL WHERE label_column IS NULL")
                self.conn.commit()
                print("Data migration completed.")

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
        self.cursor.execute('''
        SELECT * FROM experiments WHERE id = ?
        ''', (experiment_id,))

        experiment = self.cursor.fetchone()
        if experiment:
            # Check if we're using the new schema or old schema
            columns = [column[0] for column in self.cursor.description]
            result = {
                'id': experiment[0],
                'name': experiment[1],
                'date': experiment[2],
                'description': experiment[3],
                'file_path': experiment[4],
            }

            # Add remaining fields based on schema
            if 'label_column' in columns:
                label_column_index = columns.index('label_column')
                result['label_column'] = experiment[label_column_index]
            elif 'label_file_path' in columns:
                label_file_path_index = columns.index('label_file_path')
                result['label_file_path'] = experiment[label_file_path_index]

            # Add parameters
            parameters_index = columns.index('parameters')
            result['parameters'] = json.loads(experiment[parameters_index]) if experiment[parameters_index] else None

            # Add status if available
            if 'status' in columns:
                status_index = columns.index('status')
                result['status'] = experiment[status_index]

            # Add created_at
            created_at_index = columns.index('created_at')
            result['created_at'] = experiment[created_at_index]

            return result
        return None

    def list_experiments(self, active_only=True):
        """List all experiments"""
        # Check if status column exists
        self.cursor.execute("PRAGMA table_info(experiments)")
        columns = [column[1] for column in self.cursor.fetchall()]
        has_status_column = 'status' in columns

        if has_status_column and active_only:
            self.cursor.execute('''
            SELECT id, name, date, description, status FROM experiments
            WHERE status = 'active' ORDER BY date DESC
            ''')
        else:
            # If no status column or we want all experiments
            self.cursor.execute('''
            SELECT id, name, date, description FROM experiments ORDER BY date DESC
            ''')

        experiments = []
        for row in self.cursor.fetchall():
            exp = {
                'id': row[0],
                'name': row[1],
                'date': row[2],
                'description': row[3],
            }

            # Add status if available
            if has_status_column and len(row) > 4:
                exp['status'] = row[4]
            else:
                exp['status'] = 'active'  # Default status

            experiments.append(exp)

        return experiments

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

    def get_features(self, experiment_id):
        """Get features for an experiment"""
        self.cursor.execute('''
        SELECT * FROM features WHERE experiment_id = ?
        ''', (experiment_id,))

        features = []
        for feature in self.cursor.fetchall():
            features.append({
                'id': feature[0],
                'experiment_id': feature[1],
                'feature_name': feature[2],
                'value': feature[3],
                'created_at': feature[4]
            })
        return features

    def add_labels(self, experiment_id, labels_df):
        """Add labels to the experiment"""
        # Save labels to database
        for _, row in labels_df.iterrows():
            self.cursor.execute('''
            INSERT INTO labels (experiment_id, sample_id, label_value, label_type)
            VALUES (?, ?, ?, ?)
            ''', (experiment_id, str(row['sample_id']),
                  float(row['label_value']),
                  str(row.get('label_type', 'unknown'))))

        self.conn.commit()

    def get_labels(self, experiment_id):
        """Get labels for an experiment"""
        self.cursor.execute('''
        SELECT * FROM labels WHERE experiment_id = ?
        ''', (experiment_id,))

        labels = []
        for label in self.cursor.fetchall():
            labels.append({
                'id': label[0],
                'experiment_id': label[1],
                'sample_id': label[2],
                'label_value': label[3],
                'label_type': label[4],
                'created_at': label[5]
            })
        return labels

    def close(self):
        """Close database connection"""
        self.conn.close()