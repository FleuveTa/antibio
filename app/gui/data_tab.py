from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QFileDialog, QCheckBox,
                             QGroupBox, QGridLayout, QLineEdit, QTextEdit,
                             QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime

from app.core.data_loader import load_data, preprocess_data
from app.core.experiment_manager import ExperimentManager


class DataImportTab(QWidget):
    data_loaded = pyqtSignal(bool)

    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.experiment_manager = ExperimentManager()
        self.current_experiment_id = None
        self.is_voltammetry_format = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Experiment Selection Section
        exp_select_group = QGroupBox("Experiment Selection")
        exp_select_layout = QHBoxLayout()

        self.new_exp_radio = QRadioButton("New Experiment")
        self.new_exp_radio.setChecked(True)
        self.new_exp_radio.toggled.connect(self.toggle_experiment_mode)
        exp_select_layout.addWidget(self.new_exp_radio)

        self.load_exp_radio = QRadioButton("Load Existing Experiment")
        self.load_exp_radio.toggled.connect(self.toggle_experiment_mode)
        exp_select_layout.addWidget(self.load_exp_radio)

        self.exp_combo = QComboBox()
        self.exp_combo.setEnabled(False)
        self.exp_combo.currentIndexChanged.connect(self.experiment_selected)
        exp_select_layout.addWidget(self.exp_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self.load_experiments)
        exp_select_layout.addWidget(self.refresh_btn)

        exp_select_group.setLayout(exp_select_layout)
        layout.addWidget(exp_select_group)

        # Experiment Info Section
        self.exp_group = QGroupBox("Experiment Information")
        exp_layout = QGridLayout()

        # Experiment name
        exp_layout.addWidget(QLabel("Experiment Name:"), 0, 0)
        self.exp_name = QLineEdit()
        exp_layout.addWidget(self.exp_name, 0, 1)

        # Experiment description
        exp_layout.addWidget(QLabel("Description:"), 1, 0)
        self.exp_desc = QTextEdit()
        self.exp_desc.setMaximumHeight(60)
        exp_layout.addWidget(self.exp_desc, 1, 1)

        self.exp_group.setLayout(exp_layout)
        layout.addWidget(self.exp_group)

        # File Import Section
        import_group = QGroupBox("Import Voltammetric Data")
        import_layout = QGridLayout()

        # Data source selection
        import_layout.addWidget(QLabel("Data Source:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["CSV File", "Excel File"])
        import_layout.addWidget(self.source_combo, 0, 1)

        # File browser for data
        self.file_path = QLabel("No file selected")
        import_layout.addWidget(self.file_path, 1, 0)
        self.browse_btn = QPushButton("Browse Data...")
        self.browse_btn.clicked.connect(self.browse_files)
        import_layout.addWidget(self.browse_btn, 1, 1)

        # Label column selection
        import_layout.addWidget(QLabel("Label Column:"), 2, 0)
        self.label_column = QComboBox()
        self.label_column.setEnabled(False)  # Initially disabled until file is loaded
        import_layout.addWidget(self.label_column, 2, 1)

        # Import button
        self.import_btn = QPushButton("Import Data")
        self.import_btn.clicked.connect(self.import_data)
        import_layout.addWidget(self.import_btn, 3, 0, 1, 2)

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        # Data Visualization
        viz_group = QGroupBox("Voltammetric Data Visualization")
        viz_layout = QVBoxLayout()

        # Controls for visualization
        viz_controls = QHBoxLayout()

        # Scan selection for voltammetry data
        viz_controls.addWidget(QLabel("Select Scan:"))
        self.scan_combo = QComboBox()
        self.scan_combo.setEnabled(False)
        self.scan_combo.currentIndexChanged.connect(lambda: self.plot_data("raw"))
        viz_controls.addWidget(self.scan_combo)

        # Plot type selection
        viz_controls.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line Plot", "Scatter Plot", "Line + Markers", "Step Plot"])
        self.plot_type_combo.currentIndexChanged.connect(lambda: self.plot_data("raw"))
        viz_controls.addWidget(self.plot_type_combo)

        # Plot style selection
        viz_controls.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Default", "Dark", "Colorful", "Minimal", "Scientific"])
        self.style_combo.currentIndexChanged.connect(self.update_plot_style)
        viz_controls.addWidget(self.style_combo)

        viz_layout.addLayout(viz_controls)

        # Matplotlib figure with enhanced styling
        plt.style.use('seaborn-v0_8-darkgrid')  # Use a nicer style
        self.figure = plt.figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # Add navigation toolbar for zoom/pan functionality
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add toolbar and canvas to layout
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)

        # Plot type buttons
        plot_layout = QHBoxLayout()
        self.raw_btn = QPushButton("Cyclic Voltammogram")
        self.raw_btn.clicked.connect(lambda: self.plot_data("raw"))
        plot_layout.addWidget(self.raw_btn)

        self.box_btn = QPushButton("Box Plot")
        self.box_btn.clicked.connect(lambda: self.plot_data("box"))
        plot_layout.addWidget(self.box_btn)

        viz_layout.addLayout(plot_layout)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Next Steps Section
        next_steps_group = QGroupBox("Next Steps")
        next_steps_layout = QVBoxLayout()

        next_steps_label = QLabel("After importing data, proceed to the Preprocessing tab to:")
        next_steps_layout.addWidget(next_steps_label)

        steps_list = QLabel(
            "• Apply normalization and outlier removal\n"
            "• Perform signal smoothing and baseline correction\n"
            "• Detect peaks in the voltammetry data\n"
            "• Visualize the processed data"
        )
        steps_list.setStyleSheet("margin-left: 20px;")
        next_steps_layout.addWidget(steps_list)

        next_steps_group.setLayout(next_steps_layout)
        layout.addWidget(next_steps_group)

        self.setLayout(layout)

    def browse_files(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self.load_columns(file_path)

    def load_columns(self, file_path):
        """Load column names from the selected file and populate the label column dropdown"""
        try:
            # Get skip rows from input
            skip_rows = self.parse_skip_rows(self.skip_rows_input.text())
            if skip_rows:
                print(f"Skipping rows: {skip_rows}")

            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                # Try with different delimiters
                try:
                    df = pd.read_csv(file_path, skiprows=skip_rows if skip_rows else None)
                except:
                    df = pd.read_csv(file_path, sep=';', skiprows=skip_rows if skip_rows else None)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, skiprows=skip_rows if skip_rows else None)
            else:
                raise ValueError("Unsupported file format")

            # Check if this is the special voltammetry format (columns are voltage values)
            is_voltammetry_format = False
            voltage_columns = []

            # Check if column headers can be converted to float or represent voltage values
            for col in df.columns:
                try:
                    # Try to convert to float directly
                    try:
                        float(col)
                        voltage_columns.append(col)
                        is_voltammetry_format = True
                    except ValueError:
                        # Check if it's a complex voltage format like "-0.795.1925"
                        # This could be a voltage with additional information
                        if isinstance(col, str) and col.replace('-', '').replace('.', '').isdigit():
                            # It has only digits, minus signs, and dots - likely a voltage
                            # For parsing, we'll take the first part before the second dot
                            parts = col.split('.')
                            if len(parts) >= 2:
                                # Try to parse as "major.minor" format
                                try:
                                    voltage_value = float(f"{parts[0]}.{parts[1]}")
                                    voltage_columns.append(col)
                                    is_voltammetry_format = True
                                except ValueError:
                                    pass
                except (TypeError):
                    pass

            # Print debug information
            print(f"Column headers: {df.columns.tolist()}")
            print(f"Detected voltage columns: {voltage_columns}")

            if is_voltammetry_format and len(voltage_columns) > 0:
                # For voltammetry format, we don't need to select a label column from the data
                # since we'll transform it to Potential/Current format
                QMessageBox.information(self, "Voltammetry Format Detected",
                                      f"Detected voltammetry format with {len(voltage_columns)} voltage columns. \n\n"
                                      f"The data will be automatically converted to Potential/Current format.")

                # Store the dataframe temporarily
                self.temp_df = df
                self.is_voltammetry_format = True

                # Update label column dropdown with any non-voltage columns that might contain labels
                self.label_column.clear()
                self.label_column.addItem("None (No Labels)")

                # Add first column (if it's not a voltage) as a potential label column
                if len(df.columns) > 0 and df.columns[0] not in voltage_columns:
                    self.label_column.addItem(df.columns[0])

                # Enable the dropdown
                self.label_column.setEnabled(True)

            else:
                # Standard format - store the dataframe temporarily
                self.temp_df = df
                self.is_voltammetry_format = False

                # Update label column dropdown
                self.label_column.clear()
                self.label_column.addItem("None (No Labels)")

                # Add all columns to the dropdown
                for col in df.columns:
                    if col not in ["Potential", "Current"]:
                        self.label_column.addItem(col)

                # Enable the dropdown
                self.label_column.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading columns: {str(e)}")

    def toggle_experiment_mode(self, checked):
        """Toggle between new experiment and load experiment modes"""
        if self.new_exp_radio.isChecked():
            # New experiment mode
            self.exp_group.setEnabled(True)
            self.exp_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            self.browse_btn.setEnabled(True)
            self.label_column.setEnabled(True)
            self.import_btn.setText("Import Data")
        else:
            # Load experiment mode
            self.exp_group.setEnabled(False)
            self.exp_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.browse_btn.setEnabled(False)
            self.label_column.setEnabled(False)
            self.import_btn.setText("Load Experiment")
            self.load_experiments()

    def load_experiments(self):
        """Load list of experiments from database"""
        try:
            # Clear the combo box
            self.exp_combo.clear()

            # Get experiments from database
            experiments = self.experiment_manager.list_experiments()

            # Add experiments to combo box
            for exp in experiments:
                self.exp_combo.addItem(f"{exp['name']} ({exp['date']})", exp['id'])

            if not experiments:
                self.exp_combo.addItem("No experiments found")
                self.exp_combo.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading experiments: {str(e)}")

    def experiment_selected(self, index):
        """Handle experiment selection from combo box"""
        if index < 0 or self.exp_combo.itemText(index) == "No experiments found":
            return

        # Get experiment ID from combo box
        experiment_id = self.exp_combo.itemData(index)
        if experiment_id is None:
            return

        try:
            # Get experiment details
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if experiment:
                # Display experiment info
                self.exp_name.setText(experiment['name'])
                self.exp_desc.setText(experiment.get('description', ''))
                self.file_path.setText(experiment['file_path'])

                # Store experiment ID
                self.current_experiment_id = experiment_id

                # Store the current experiment ID in app_data so it's accessible to other tabs
                self.app_data["current_experiment_id"] = self.current_experiment_id
                print(f"Stored current experiment ID in app_data: {self.current_experiment_id}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading experiment details: {str(e)}")

    def import_data(self):
        """Import data or load existing experiment"""
        if self.new_exp_radio.isChecked():
            # New experiment mode
            self.create_new_experiment()
        else:
            # Load experiment mode
            self.load_existing_experiment()

    def create_new_experiment(self):
        """Create a new experiment with imported data"""
        file_path = self.file_path.text()

        if file_path == "No file selected":
            QMessageBox.warning(self, "Warning", "Please select a data file first")
            return

        try:
            # Create new experiment
            exp_name = self.exp_name.text() or f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            exp_desc = self.exp_desc.toPlainText()

            # Get selected label column
            selected_label = self.label_column.currentText()
            if selected_label == "None (No Labels)":
                selected_label = None

            # Get skip rows from input
            skip_rows = self.parse_skip_rows(self.skip_rows_input.text())
            if skip_rows:
                print(f"Skipping rows: {skip_rows}")

            # First, load and transform the data
            if hasattr(self, 'temp_df') and self.is_voltammetry_format:
                # For voltammetry format, we need to transform the data
                print("Using voltammetry format data transformation")
                # If we're using a previously loaded dataframe, make sure it was loaded with the correct skip rows
                if skip_rows and not hasattr(self, 'used_skip_rows') or self.used_skip_rows != skip_rows:
                    # Reload the data with the correct skip rows
                    print("Reloading data with skip rows")
                    if file_path.endswith('.csv'):
                        try:
                            self.temp_df = pd.read_csv(file_path, skiprows=skip_rows)
                        except:
                            self.temp_df = pd.read_csv(file_path, sep=';', skiprows=skip_rows)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        self.temp_df = pd.read_excel(file_path, skiprows=skip_rows)
                    self.used_skip_rows = skip_rows

                df = self.transform_voltammetry_data(self.temp_df)
            elif hasattr(self, 'temp_df'):
                # Use the already loaded dataframe for standard format
                # Check if we need to reload with different skip rows
                if skip_rows and not hasattr(self, 'used_skip_rows') or self.used_skip_rows != skip_rows:
                    # Reload the data with the correct skip rows
                    print("Reloading data with skip rows")
                    if file_path.endswith('.csv'):
                        try:
                            self.temp_df = pd.read_csv(file_path, skiprows=skip_rows)
                        except:
                            self.temp_df = pd.read_csv(file_path, sep=';', skiprows=skip_rows)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        self.temp_df = pd.read_excel(file_path, skiprows=skip_rows)
                    self.used_skip_rows = skip_rows

                df = self.temp_df.copy()
            else:
                # Load the data file with skip rows
                if file_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path, skiprows=skip_rows if skip_rows else None)
                    except:
                        df = pd.read_csv(file_path, sep=';', skiprows=skip_rows if skip_rows else None)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path, skiprows=skip_rows if skip_rows else None)
                else:
                    df = load_data(file_path)  # Fallback to the original loader

            # Create the experiment after we have the data
            self.current_experiment_id = self.experiment_manager.create_experiment(
                name=exp_name,
                file_path=file_path,
                label_column=selected_label,
                description=exp_desc
            )

            # Check if the dataset has the required columns
            if "Potential" not in df.columns or "Current" not in df.columns:
                raise ValueError("The data file must contain 'Potential' and 'Current' columns for voltammetric analysis")

            # Extract labels if a label column is selected
            if selected_label and selected_label in df.columns:
                # Create a labels dataframe
                labels_df = pd.DataFrame({
                    'sample_id': range(len(df)),
                    'label_value': df[selected_label]
                })

                # Store labels in app_data
                self.app_data["labels"] = labels_df

                # Add labels to experiment
                self.experiment_manager.add_labels(self.current_experiment_id, labels_df)

                QMessageBox.information(self, "Success", f"Extracted {len(labels_df)} labels from column '{selected_label}'")

                # Remove label column from the dataset to avoid confusion
                df = df.drop(columns=[selected_label])

            # Store the dataset
            self.app_data["dataset"] = df

            # Store the current experiment ID in app_data so it's accessible to other tabs
            self.app_data["current_experiment_id"] = self.current_experiment_id
            print(f"Stored current experiment ID in app_data: {self.current_experiment_id}")

            # If this is voltammetry format data, populate the scan combo box
            if self.is_voltammetry_format and "voltammetry_row_indices" in self.app_data:
                self.scan_combo.clear()
                # Only add individual scans, no "All Scans" option to prevent lag
                for idx in self.app_data["voltammetry_row_indices"]:
                    self.scan_combo.addItem(f"Scan {idx}")
                self.scan_combo.setEnabled(True)
                # Default to the first scan if available
                if self.scan_combo.count() > 0:
                    self.scan_combo.setCurrentIndex(0)

            # Plot raw data by default
            self.plot_data("raw")

            # Signal that data is loaded and ready for preprocessing
            self.data_loaded.emit(True)

            QMessageBox.information(self, "Success", f"Created new experiment: {exp_name} (ID: {self.current_experiment_id})")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating experiment: {str(e)}")

    def transform_voltammetry_data(self, df):
        """Transform voltammetry data from wide format (voltages as columns) to long format (Potential and Current columns)"""
        # Check if column headers can be converted to float or represent voltage values
        voltage_columns = []
        voltage_values = {}  # Dictionary to store the actual voltage value for each column

        for col in df.columns:
            try:
                # Try to convert to float directly
                try:
                    voltage_value = float(col)
                    voltage_columns.append(col)
                    voltage_values[col] = voltage_value
                except ValueError:
                    # Check if it's a complex voltage format like "-0.795.1925"
                    if isinstance(col, str) and col.replace('-', '').replace('.', '').isdigit():
                        # It has only digits, minus signs, and dots - likely a voltage
                        # For parsing, we'll take the first part before the second dot
                        parts = col.split('.')
                        if len(parts) >= 2:
                            # Try to parse as "major.minor" format
                            try:
                                voltage_value = float(f"{parts[0]}.{parts[1]}")
                                voltage_columns.append(col)
                                voltage_values[col] = voltage_value
                            except ValueError:
                                pass
            except (TypeError):
                pass

        if len(voltage_columns) == 0:
            raise ValueError("No voltage columns found in the data")

        print(f"Found {len(voltage_columns)} voltage columns")

        # Create a new dataframe with Potential and Current columns
        new_data = []

        # Store row indices for later selection
        row_indices = []

        # For each row in the original dataframe
        for idx, row in df.iterrows():
            row_data = []
            # For each voltage column
            for voltage_col in voltage_columns:
                # Add a row with Potential = voltage value, Current = cell value
                try:
                    current_value = row[voltage_col]
                    if pd.notna(current_value):  # Skip NaN values
                        # Use the parsed voltage value from our dictionary
                        potential_value = voltage_values[voltage_col]
                        row_data.append({
                            'Potential': potential_value,
                            'Current': float(current_value),
                            'RowIndex': idx  # Store the original row index
                        })
                except (ValueError, TypeError) as e:
                    print(f"Error converting value at row {idx}, column {voltage_col}: {e}")
                    # Skip cells that can't be converted to float
                    pass

            # If we have data for this row, add it to our dataset and track the row index
            if row_data:
                new_data.extend(row_data)
                if idx not in row_indices:
                    row_indices.append(idx)

        if len(new_data) == 0:
            raise ValueError("No valid data points found after transformation")

        # Create new dataframe
        new_df = pd.DataFrame(new_data)
        print(f"Transformed data: {len(new_df)} data points, columns: {new_df.columns.tolist()}")

        # Store row indices in app_data for later use in visualization
        self.app_data["voltammetry_row_indices"] = row_indices
        print(f"Found {len(row_indices)} voltammetry scans (rows)")

        return new_df

    def load_existing_experiment(self):
        """Load an existing experiment"""
        if self.current_experiment_id is None:
            QMessageBox.warning(self, "Warning", "Please select an experiment first")
            return

        try:
            # Get experiment details
            experiment = self.experiment_manager.get_experiment(self.current_experiment_id)
            if not experiment:
                raise ValueError(f"Experiment with ID {self.current_experiment_id} not found")

            # Load the data file
            file_path = experiment['file_path']
            raw_df = load_data(file_path)

            # Check if this is an old experiment format (without RowIndex)
            if "RowIndex" not in raw_df.columns and "Potential" in raw_df.columns and "Current" in raw_df.columns:
                print("Detected old experiment format, transforming data...")
                # We need to transform the data to match our new format
                # First, detect if this is voltammetry data by checking the file
                self.is_voltammetry_format = True  # Assume it's voltammetry data for old experiments

                # Load the original file to get the row structure
                try:
                    # Try loading with comma separator first
                    original_df = pd.read_csv(file_path)
                except:
                    # Try with semicolon separator
                    original_df = pd.read_csv(file_path, sep=';')

                # Transform the data using our new method
                transformed_df = self.transform_voltammetry_data(original_df)

                # Store the transformed dataset
                self.app_data["dataset"] = transformed_df
                print(f"Transformed data: {len(transformed_df)} data points")
            else:
                # This is either a new format experiment or not voltammetry data
                self.app_data["dataset"] = raw_df

            # Load labels if available
            labels = self.experiment_manager.get_labels(self.current_experiment_id)
            if labels:
                # Convert to DataFrame
                labels_df = pd.DataFrame(labels)
                self.app_data["labels"] = labels_df
                QMessageBox.information(self, "Success", f"Loaded {len(labels_df)} labels")

            # Load preprocessing steps
            preprocessing_steps = self.experiment_manager.get_preprocessing_steps(self.current_experiment_id)
            if preprocessing_steps:
                self.app_data["preprocessing_steps"] = preprocessing_steps
                QMessageBox.information(self, "Success", f"Loaded {len(preprocessing_steps)} preprocessing steps")

            # Store the current experiment ID in app_data so it's accessible to other tabs
            self.app_data["current_experiment_id"] = self.current_experiment_id
            print(f"Stored current experiment ID in app_data: {self.current_experiment_id}")

            # Update status and plot raw data by default
            self.plot_data("raw")

            # Signal that data is loaded and ready for preprocessing
            self.data_loaded.emit(True)

            QMessageBox.information(self, "Success", f"Loaded experiment: {experiment['name']}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading experiment: {str(e)}")

    def plot_data(self, plot_type):
        if self.app_data.get("dataset") is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Use processed data if available, otherwise use original dataset
        if "processed_data" in self.app_data and self.app_data["processed_data"] is not None:
            df = self.app_data["processed_data"]
            title_prefix = "Processed "
        else:
            df = self.app_data["dataset"]
            title_prefix = ""

        if plot_type == "raw":
            # Plotting for voltammetric data

            # Always filter by scan to prevent lag
            if self.scan_combo.isEnabled() and self.scan_combo.count() > 0 and "RowIndex" in df.columns:
                selected_scan = self.scan_combo.currentText()
                # Extract the scan number from the text (format: "Scan X")
                scan_idx = int(selected_scan.split(" ")[1])
                # Filter data for the selected scan
                plot_df = df[df["RowIndex"] == scan_idx]
                title_suffix = f" - Scan {scan_idx}"
            else:
                # If no scan selection is available, just use a small subset of data
                # to prevent performance issues
                plot_df = df.head(100)  # Limit to first 100 points
                title_suffix = " (Limited View)"

            # Check if we should use line or scatter plot
            plot_style = self.plot_type_combo.currentText() if hasattr(self, 'plot_type_combo') else "Line Plot"

            # Create a more professional looking plot based on selected style
            if plot_style == "Line Plot":
                ax.plot(plot_df["Potential"], plot_df["Current"], linewidth=2, color='#1f77b4')

                # Add a subtle shadow effect for depth
                ax.plot(plot_df["Potential"], plot_df["Current"], linewidth=4, color='#1f77b4', alpha=0.2)

            elif plot_style == "Scatter Plot":
                ax.scatter(plot_df["Potential"], plot_df["Current"], s=25, color='#1f77b4',
                          alpha=0.7, edgecolors='white', linewidths=0.5)

                # Add connecting line with low alpha for better visualization
                ax.plot(plot_df["Potential"], plot_df["Current"], linewidth=1, color='#1f77b4', alpha=0.3)

            elif plot_style == "Line + Markers":
                ax.plot(plot_df["Potential"], plot_df["Current"], linewidth=2, color='#1f77b4',
                       marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#1f77b4')

            elif plot_style == "Step Plot":
                ax.step(plot_df["Potential"], plot_df["Current"], linewidth=2, color='#1f77b4', where='post')
                # Add points at the steps
                ax.scatter(plot_df["Potential"], plot_df["Current"], s=20, color='#1f77b4',
                          alpha=0.7, edgecolors='white', linewidths=0.5, zorder=10)

            # Enhanced axis labels with better font
            ax.set_xlabel("Potential (V)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Current (μA)", fontsize=12, fontweight='bold')
            ax.set_title(f"{title_prefix}Cyclic Voltammogram{title_suffix}", fontsize=14, fontweight='bold')

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Customize tick labels
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Add a light background color to the plot area
            ax.set_facecolor('#f8f9fa')

            # If peak detection was applied, mark the peaks
            if "Peaks" in plot_df.columns:
                peaks_df = plot_df[plot_df["Peaks"] == True]
                if not peaks_df.empty:
                    ax.scatter(peaks_df["Potential"], peaks_df["Current"],
                              s=100, color='red', marker='o', label='Peaks',
                              edgecolors='white', linewidths=1.5, zorder=10)

                    # Add peak annotations
                    for i, (pot, curr) in enumerate(zip(peaks_df["Potential"], peaks_df["Current"])):
                        ax.annotate(f"Peak {i+1}",
                                   xy=(pot, curr),
                                   xytext=(0, 10),
                                   textcoords='offset points',
                                   ha='center',
                                   fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

                    # Enhanced legend
                    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)

            # Add a box around the plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('#cccccc')

            # Tight layout for better spacing
            self.figure.tight_layout()

            # Add data cursor for interactive data exploration
            from matplotlib.widgets import Cursor
            self.cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

            # Connect to mouse motion events for showing data values
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

            # Add a text annotation for displaying values
            self.annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                     fontsize=9, ha='center', visible=False)

        elif plot_type == "box":
            # Box plot of the data features
            numeric_columns = df.select_dtypes(include=np.number).columns
            df[numeric_columns].boxplot(ax=ax)
            ax.set_title(f"{title_prefix}Feature Distribution")

        self.canvas.draw()

    def preprocess(self):
        if self.app_data.get("dataset") is None or self.current_experiment_id is None:
            return

        options = {
            "normalize": self.normalize_cb.isChecked(),
            "remove_outliers": self.outliers_cb.isChecked(),
            "fill_missing": self.missing_cb.isChecked(),
            "smooth": self.smooth_cb.isChecked(),
            "baseline_correction": self.baseline_cb.isChecked(),
            "peak_detection": self.peak_cb.isChecked()
        }

        try:
            # Apply preprocessing and get transformers
            self.app_data["processed_data"], transformers = preprocess_data(
                self.app_data["dataset"],
                "Voltammetric",
                options
            )

            # Save processed data
            processed_file_path = self.experiment_manager.save_processed_data(
                self.current_experiment_id,
                self.app_data["processed_data"],
                "preprocessed"
            )

            # Save transformers and record preprocessing steps
            for step_name, transformer in transformers.items():
                self.experiment_manager.add_preprocessing_step(
                    self.current_experiment_id,
                    step_name,
                    transformer=transformer
                )

            # Show a message about what was processed
            applied_options = [key for key, value in options.items() if value]
            if applied_options:
                print(f"Applied preprocessing: {', '.join(applied_options)}")
                print(f"Processed data saved to: {processed_file_path}")
                print(f"Transformers saved for: {', '.join(transformers.keys())}")
            else:
                print("No preprocessing options selected")

            # Update plot to show processed data
            self.plot_data("raw")

        except Exception as e:
            print(f"Error preprocessing data: {e}")

    def reset_processing(self):
        """Reset processed data and show original data"""
        if "processed_data" in self.app_data:
            self.app_data["processed_data"] = None
            self.plot_data("raw")
            print("Reset to original data")

    def update_plot_style(self):
        """Update the plot style based on the selected style"""
        style = self.style_combo.currentText()

        if style == "Default":
            plt.style.use('seaborn-v0_8-darkgrid')
        elif style == "Dark":
            plt.style.use('dark_background')
        elif style == "Colorful":
            plt.style.use('seaborn-v0_8-colorblind')
        elif style == "Minimal":
            plt.style.use('ggplot')
        elif style == "Scientific":
            plt.style.use('seaborn-v0_8-paper')

        # Redraw the plot with the new style
        self.plot_data("raw")

    def on_mouse_move(self, event):
        """Handle mouse movement over the plot to show data values"""
        if not event.inaxes:
            # Mouse is not over the plot
            if hasattr(self, 'annot') and self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            return

        # Get the current axes
        ax = event.inaxes

        # Get the data currently being displayed
        if "processed_data" in self.app_data and self.app_data["processed_data"] is not None:
            df = self.app_data["processed_data"]
        else:
            df = self.app_data.get("dataset")

        if df is None:
            return

        # Always filter by scan to prevent lag
        if self.scan_combo.isEnabled() and self.scan_combo.count() > 0 and "RowIndex" in df.columns:
            selected_scan = self.scan_combo.currentText()
            scan_idx = int(selected_scan.split(" ")[1])
            plot_df = df[df["RowIndex"] == scan_idx]
        else:
            # If no scan selection is available, just use a small subset of data
            plot_df = df.head(100)  # Limit to first 100 points

        # Find the closest point to the cursor
        x, y = event.xdata, event.ydata
        points = plot_df[["Potential", "Current"]].values
        if len(points) == 0:
            return

        # Calculate distances to all points
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        closest_idx = np.argmin(distances)
        closest_x, closest_y = points[closest_idx]

        # Only show annotation if we're close enough to a point
        if distances[closest_idx] < 0.05:  # Adjust threshold as needed
            # Update annotation with the data values
            self.annot.xy = (closest_x, closest_y)
            self.annot.set_text(f"Potential: {closest_x:.4f} V\nCurrent: {closest_y:.4e} μA")
            self.annot.set_visible(True)

            # Highlight the point
            if hasattr(self, 'highlight_point'):
                self.highlight_point.remove()
            self.highlight_point = ax.scatter([closest_x], [closest_y], s=100,
                                             facecolor='none', edgecolor='red', linewidth=2, zorder=10)

            self.canvas.draw_idle()
        elif hasattr(self, 'annot') and self.annot.get_visible():
            self.annot.set_visible(False)
            if hasattr(self, 'highlight_point'):
                self.highlight_point.remove()
            self.canvas.draw_idle()

    def closeEvent(self, event):
        """Clean up when the widget is closed"""
        self.experiment_manager.close()
        super().closeEvent(event)