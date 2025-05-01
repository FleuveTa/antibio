from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QCheckBox,
                             QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime

from app.core.preprocessing import preprocess_wide_data
from app.core.experiment_manager import ExperimentManager


class PreprocessingTab(QWidget):
    """
    Tab for preprocessing voltammetric data
    """
    preprocessing_done = pyqtSignal(bool)

    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.experiment_manager = ExperimentManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Visualization Section
        viz_group = QGroupBox("Data Visualization")
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
        self.figure = plt.figure(figsize=(8, 6), dpi=100)  # Larger figure size
        self.canvas = FigureCanvas(self.figure)

        # Add navigation toolbar for zoom/pan functionality
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add toolbar and canvas to layout
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Preprocessing Options
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QGridLayout()

        # Data cleaning options
        cleaning_group = QGroupBox("Data Cleaning")
        cleaning_layout = QGridLayout()

        self.missing_cb = QCheckBox("Fill missing values")
        self.missing_cb.setToolTip("Replace missing values with column means")
        cleaning_layout.addWidget(self.missing_cb, 0, 0)

        # Remove outliers option has been removed to preserve all data points
        # self.outliers_cb = QCheckBox("Remove outliers")
        # self.outliers_cb.setToolTip("Remove data points with z-score > 3")
        # cleaning_layout.addWidget(self.outliers_cb, 0, 1)

        cleaning_group.setLayout(cleaning_layout)
        preproc_layout.addWidget(cleaning_group, 0, 0, 1, 2)

        # Signal processing options
        signal_group = QGroupBox("Signal Processing")
        signal_layout = QGridLayout()

        self.smooth_cb = QCheckBox("Signal smoothing")
        self.smooth_cb.setToolTip("Apply Savitzky-Golay filter to reduce noise")
        signal_layout.addWidget(self.smooth_cb, 0, 0)

        self.baseline_cb = QCheckBox("Baseline correction")
        self.baseline_cb.setToolTip("Remove linear trend from signal")
        signal_layout.addWidget(self.baseline_cb, 0, 1)

        self.peaks_cb = QCheckBox("Detect peaks")
        self.peaks_cb.setToolTip("Detect peaks and valleys in the voltammetry data")
        self.peaks_cb.setChecked(True)  # Enable by default
        signal_layout.addWidget(self.peaks_cb, 1, 0)

        signal_group.setLayout(signal_layout)
        preproc_layout.addWidget(signal_group, 1, 0, 1, 2)

        # Placeholder for future options if needed
        # We removed normalization as it's not necessary and can affect electrochemical relationships

        # Button layout
        btn_layout = QHBoxLayout()

        self.preprocess_btn = QPushButton("Apply Preprocessing")
        self.preprocess_btn.clicked.connect(self.preprocess)
        self.preprocess_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.preprocess_btn)

        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.clicked.connect(self.reset_processing)
        self.reset_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.reset_btn)

        preproc_layout.addLayout(btn_layout, 3, 0, 1, 2)

        preproc_group.setLayout(preproc_layout)
        layout.addWidget(preproc_group)

        self.setLayout(layout)

        # Initially disable preprocessing button until data is loaded
        self.preprocess_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def showEvent(self, event):
        """Handle when tab is shown"""
        super().showEvent(event)

        # Check if original wide data is loaded and update UI accordingly
        if "original_wide_data" in self.app_data and self.app_data["original_wide_data"] is not None:
            self.preprocess_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            print("Found original_wide_data for preprocessing")

            # Update scan combo box if voltammetry data is loaded
            if "voltammetry_row_indices" in self.app_data:
                self.scan_combo.clear()
                # Only add individual scans, no "All Scans" option to prevent lag
                for idx in self.app_data["voltammetry_row_indices"]:
                    self.scan_combo.addItem(f"Scan {idx}")
                self.scan_combo.setEnabled(True)
                # Default to the first scan if available
                if self.scan_combo.count() > 0:
                    self.scan_combo.setCurrentIndex(0)

            # Check if there are preprocessing steps loaded from an existing experiment
            if "preprocessing_steps" in self.app_data and self.app_data["preprocessing_steps"]:
                # Set checkboxes based on loaded preprocessing steps
                steps = [step["name"] for step in self.app_data["preprocessing_steps"]]
                # self.normalize_cb.setChecked("normalize" in steps)  # Normalization removed
                # self.outliers_cb.setChecked("remove_outliers" in steps)  # Remove outliers option has been removed
                self.missing_cb.setChecked("fill_missing" in steps)
                self.smooth_cb.setChecked("smooth" in steps)
                self.baseline_cb.setChecked("baseline_correction" in steps)
                self.peaks_cb.setChecked("detect_peaks" in steps)  # Fixed variable name and step name

                # If processed data was saved, try to load it
                current_experiment_id = self.app_data.get("current_experiment_id")
                if current_experiment_id:
                    try:
                        processed_data = self.experiment_manager.get_processed_data(current_experiment_id)
                        if processed_data is not None:
                            # Check if this is an old format (without RowIndex)
                            if "RowIndex" not in processed_data.columns and "Potential" in processed_data.columns and "Current" in processed_data.columns:
                                print("Detected old processed data format, adding row indices...")
                                # Add RowIndex based on the dataset's row indices
                                if "dataset" in self.app_data and "voltammetry_row_indices" in self.app_data:
                                    # Get the first row index (usually 0)
                                    first_row_idx = self.app_data["voltammetry_row_indices"][0] if self.app_data["voltammetry_row_indices"] else 0
                                    # Add RowIndex column with the first row index
                                    processed_data["RowIndex"] = first_row_idx
                                    print(f"Added RowIndex {first_row_idx} to processed data")

                            self.app_data["processed_data"] = processed_data
                            print("Loaded processed data from experiment")
                    except Exception as e:
                        print(f"Could not load processed data: {e}")

            # Plot the data
            self.plot_data("raw")

    def plot_data(self, plot_type):
        # Check if we have data to plot (either processed_wide_data, original_wide_data, or dataset as fallback)
        if not ("processed_wide_data" in self.app_data or "original_wide_data" in self.app_data or "dataset" in self.app_data):
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Use processed wide data if available, otherwise use original wide data, or dataset as fallback
        if "processed_wide_data" in self.app_data and self.app_data["processed_wide_data"] is not None:
            df = self.app_data["processed_wide_data"]
            title_prefix = "Processed "
        elif "original_wide_data" in self.app_data and self.app_data["original_wide_data"] is not None:
            df = self.app_data["original_wide_data"]
            title_prefix = "Original "
        else:
            df = self.app_data["dataset"]
            title_prefix = ""

        if plot_type == "raw":
            # Plotting for voltammetric data

            # Check if we're working with wide format data (voltage columns) or long format data (Potential/Current columns)
            if "Potential" in df.columns and "Current" in df.columns:
                # Long format data - use existing code
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

                # Extract x and y data for plotting
                x_data = plot_df["Potential"]
                y_data = plot_df["Current"]
            else:
                # Wide format data - voltage columns
                # Select the first row (first scan) for plotting
                if self.scan_combo.isEnabled() and self.scan_combo.count() > 0:
                    selected_scan = self.scan_combo.currentText()
                    # Extract the scan number from the text (format: "Scan X")
                    scan_idx = int(selected_scan.split(" ")[1])
                    # If scan_idx is within range of dataframe rows
                    if scan_idx < len(df):
                        # Select the row corresponding to the scan
                        plot_df = df.iloc[[scan_idx]]
                    else:
                        # Default to first row
                        plot_df = df.iloc[[0]]
                    title_suffix = f" - Scan {scan_idx}"
                else:
                    # Default to first row
                    plot_df = df.iloc[[0]]
                    title_suffix = " - Scan 0"

                # Extract voltage columns (numeric columns that are not metadata)
                # Add 'Unnamed: 0' and other common non-voltage columns to the metadata list
                metadata_cols = ['concentration', 'antibiotic', 'label', 'class', 'target', 'path', 'row_index',
                               'unnamed: 0', 'index', 'id', 'sample_id', 'scan_id']

                # Filter out columns that are likely not voltage values
                voltage_cols = []
                for col in plot_df.columns:
                    # Skip metadata columns (case insensitive)
                    if any(meta.lower() in str(col).lower() for meta in metadata_cols):
                        continue

                    # Try to convert to float to check if it's a numeric column
                    try:
                        # For columns like '-0.795.1925', take only the first part
                        if isinstance(col, str) and col.count('.') > 1:
                            parts = col.split('.')
                            # Try to parse as "major.minor" format
                            float(f"{parts[0]}.{parts[1]}")
                        else:
                            float(col)
                        # If conversion succeeds, it's likely a voltage column
                        voltage_cols.append(col)
                    except (ValueError, TypeError):
                        # Not a voltage column, skip it
                        continue

                print(f"Found {len(voltage_cols)} voltage columns for plotting")

                # Sort voltage columns by their float values
                try:
                    # For columns with multiple dots, use only the first part for sorting
                    def get_sort_value(col):
                        if isinstance(col, str) and col.count('.') > 1:
                            parts = col.split('.')
                            return float(f"{parts[0]}.{parts[1]}")
                        return float(col)

                    voltage_cols = sorted(voltage_cols, key=get_sort_value)
                except Exception as e:
                    # If sorting fails, just use the columns as they are
                    print(f"Warning: Could not sort voltage columns: {e}")

                # Extract x and y data for plotting
                x_data = []
                y_data = []

                # Process each voltage column
                for col in voltage_cols:
                    try:
                        # Convert column name to float for x-axis
                        if isinstance(col, str) and col.count('.') > 1:
                            parts = col.split('.')
                            x_val = float(f"{parts[0]}.{parts[1]}")
                        else:
                            x_val = float(col)

                        # Get the current value for this voltage
                        y_val = plot_df[col].values[0]

                        # Add to our data arrays
                        x_data.append(x_val)
                        y_data.append(y_val)
                    except Exception as e:
                        print(f"Warning: Could not process column {col}: {e}")
                        continue

                print(f"Prepared {len(x_data)} data points for plotting")

            # Check if we should use line or scatter plot
            plot_style = self.plot_type_combo.currentText() if hasattr(self, 'plot_type_combo') else "Line Plot"

            # Create a more professional looking plot based on selected style
            if plot_style == "Line Plot":
                ax.plot(x_data, y_data, linewidth=2, color='#1f77b4')

                # Add a subtle shadow effect for depth
                ax.plot(x_data, y_data, linewidth=4, color='#1f77b4', alpha=0.2)

                # Plot peaks if available (only for long format data)
                if "Potential" in df.columns and "Current" in df.columns and "Peaks" in plot_df.columns:
                    peak_points = plot_df[plot_df["Peaks"] == True]
                    if not peak_points.empty:
                        ax.scatter(peak_points["Potential"], peak_points["Current"],
                                  s=80, color='red', marker='o', alpha=0.7,
                                  edgecolors='white', linewidths=1, zorder=10,
                                  label='Detected Peaks')

            elif plot_style == "Scatter Plot":
                ax.scatter(x_data, y_data, s=25, color='#1f77b4',
                          alpha=0.7, edgecolors='white', linewidths=0.5)

                # Add connecting line with low alpha for better visualization
                ax.plot(x_data, y_data, linewidth=1, color='#1f77b4', alpha=0.3)

                # Plot peaks if available (only for long format data)
                if "Potential" in df.columns and "Current" in df.columns and "Peaks" in plot_df.columns:
                    peak_points = plot_df[plot_df["Peaks"] == True]
                    if not peak_points.empty:
                        ax.scatter(peak_points["Potential"], peak_points["Current"],
                                  s=80, color='red', marker='o', alpha=0.7,
                                  edgecolors='white', linewidths=1, zorder=10,
                                  label='Detected Peaks')

            elif plot_style == "Line + Markers":
                ax.plot(x_data, y_data, linewidth=2, color='#1f77b4',
                       marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#1f77b4')

                # Plot peaks if available (only for long format data)
                if "Potential" in df.columns and "Current" in df.columns and "Peaks" in plot_df.columns:
                    peak_points = plot_df[plot_df["Peaks"] == True]
                    if not peak_points.empty:
                        ax.scatter(peak_points["Potential"], peak_points["Current"],
                                  s=80, color='red', marker='*', alpha=0.9,
                                  edgecolors='white', linewidths=1, zorder=10,
                                  label='Detected Peaks')

            elif plot_style == "Step Plot":
                ax.step(x_data, y_data, linewidth=2, color='#1f77b4', where='post')
                # Add points at the steps
                ax.scatter(x_data, y_data, s=20, color='#1f77b4',
                          alpha=0.7, edgecolors='white', linewidths=0.5, zorder=10)

                # Plot peaks if available (only for long format data)
                if "Potential" in df.columns and "Current" in df.columns and "Peaks" in plot_df.columns:
                    peak_points = plot_df[plot_df["Peaks"] == True]
                    if not peak_points.empty:
                        ax.scatter(peak_points["Potential"], peak_points["Current"],
                                  s=80, color='red', marker='D', alpha=0.8,
                                  edgecolors='white', linewidths=1, zorder=11,
                                  label='Detected Peaks')

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

            # Add legend if needed
            if plot_style in ["Line + Markers", "Scatter Plot"]:
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
        if "processed_wide_data" in self.app_data and self.app_data["processed_wide_data"] is not None:
            df = self.app_data["processed_wide_data"]
        elif "original_wide_data" in self.app_data and self.app_data["original_wide_data"] is not None:
            df = self.app_data["original_wide_data"]
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

        # Check if we're working with wide format data (voltage columns) or long format data (Potential/Current columns)
        if "Potential" in df.columns and "Current" in df.columns:
            # Long format data
            points = plot_df[["Potential", "Current"]].values
        else:
            # Wide format data - voltage columns
            # Add 'Unnamed: 0' and other common non-voltage columns to the metadata list
            metadata_cols = ['concentration', 'antibiotic', 'label', 'class', 'target', 'path', 'row_index',
                           'unnamed: 0', 'index', 'id', 'sample_id', 'scan_id']

            # Filter out columns that are likely not voltage values
            voltage_cols = []
            for col in plot_df.columns:
                # Skip metadata columns (case insensitive)
                if any(meta.lower() in str(col).lower() for meta in metadata_cols):
                    continue

                # Try to convert to float to check if it's a numeric column
                try:
                    # For columns like '-0.795.1925', take only the first part
                    if isinstance(col, str) and col.count('.') > 1:
                        parts = col.split('.')
                        # Try to parse as "major.minor" format
                        float(f"{parts[0]}.{parts[1]}")
                    else:
                        float(col)
                    # If conversion succeeds, it's likely a voltage column
                    voltage_cols.append(col)
                except (ValueError, TypeError):
                    # Not a voltage column, skip it
                    continue

            # Create points array with x values (voltage) and y values (current)
            x_values = []
            y_values = []

            # Process each voltage column
            for col in voltage_cols:
                try:
                    # Convert column name to float for x-axis
                    if isinstance(col, str) and col.count('.') > 1:
                        parts = col.split('.')
                        x_val = float(f"{parts[0]}.{parts[1]}")
                    else:
                        x_val = float(col)

                    # Get the current value for this voltage
                    y_val = plot_df[col].values[0]

                    # Add to our data arrays
                    x_values.append(x_val)
                    y_values.append(y_val)
                except Exception as e:
                    # Skip this column if there's an error
                    continue

            # Create the points array
            if len(x_values) > 0:
                points = np.column_stack((x_values, y_values))
            else:
                # If no valid points, return an empty array
                points = np.array([])

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

    def preprocess(self):
        """Apply preprocessing to the data"""
        # Check if we have original wide data to process
        if "original_wide_data" in self.app_data and self.app_data["original_wide_data"] is not None:
            # Use the original wide format data directly
            data = self.app_data["original_wide_data"]
            print(f"Found original_wide_data with shape: {data.shape}")
            print(f"Using original_wide_data for preprocessing")
        else:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import data first.")
            return

        # Get current experiment ID from app_data
        current_experiment_id = self.app_data.get("current_experiment_id")
        if current_experiment_id is None:
            QMessageBox.warning(self, "Warning", "No active experiment. Please create or load an experiment first.")
            return

        options = {
            "normalize": False,  # Normalization removed as it's not necessary for electrochemical data
            "remove_outliers": False,  # Remove outliers option has been disabled to preserve all data points
            "fill_missing": self.missing_cb.isChecked(),
            "smooth": self.smooth_cb.isChecked(),
            "baseline_correction": self.baseline_cb.isChecked(),
            "detect_peaks": self.peaks_cb.isChecked()  # Add peak detection option
        }

        try:
            # Apply preprocessing directly on wide format data and get transformers
            processed_wide_data, transformers = preprocess_wide_data(
                data,
                options
            )

            # Store the processed wide data
            self.app_data["processed_wide_data"] = processed_wide_data
            print(f"Stored processed_wide_data with shape: {processed_wide_data.shape}")

            # We're working directly with wide format data, no need to convert to long format
            # Just use the processed wide data directly for visualization
            print(f"Using processed_wide_data directly for visualization")

            # Save processed data
            processed_file_path = self.experiment_manager.save_processed_data(
                current_experiment_id,
                self.app_data["processed_wide_data"],
                "preprocessed"
            )

            # Save transformers and record preprocessing steps
            for step_name, transformer in transformers.items():
                self.experiment_manager.add_preprocessing_step(
                    current_experiment_id,
                    step_name,
                    transformer=transformer
                )

            # Show a message about what was processed
            applied_options = [key for key, value in options.items() if value]
            if applied_options:
                message = f"Applied preprocessing: {', '.join(applied_options)}\n"
                message += f"Processed data saved to: {processed_file_path}\n"
                message += f"Transformers saved for: {', '.join(transformers.keys())}"
                QMessageBox.information(self, "Preprocessing Complete", message)
            else:
                QMessageBox.warning(self, "No Options Selected", "No preprocessing options were selected.")

            # Update plot to show processed data
            self.plot_data("raw")

            # Signal that preprocessing is done
            self.preprocessing_done.emit(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preprocessing data: {str(e)}")

    def reset_processing(self):
        """Reset processed data and show original data"""
        if "processed_wide_data" in self.app_data:
            self.app_data["processed_wide_data"] = None
            self.plot_data("raw")
            QMessageBox.information(self, "Reset", "Reset to original data")
