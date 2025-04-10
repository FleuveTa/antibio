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

from app.core.data_loader import preprocess_data
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

        self.normalize_cb = QCheckBox("Normalize data")
        preproc_layout.addWidget(self.normalize_cb, 0, 0)

        self.outliers_cb = QCheckBox("Remove outliers")
        preproc_layout.addWidget(self.outliers_cb, 0, 1)

        self.missing_cb = QCheckBox("Fill missing values")
        preproc_layout.addWidget(self.missing_cb, 1, 0)

        self.smooth_cb = QCheckBox("Signal smoothing")
        preproc_layout.addWidget(self.smooth_cb, 1, 1)

        self.baseline_cb = QCheckBox("Baseline correction")
        preproc_layout.addWidget(self.baseline_cb, 2, 0)

        self.peak_cb = QCheckBox("Peak detection")
        preproc_layout.addWidget(self.peak_cb, 2, 1)

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

        # Check if data is loaded and update UI accordingly
        if "dataset" in self.app_data and self.app_data["dataset"] is not None:
            self.preprocess_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)

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
                self.normalize_cb.setChecked("normalize" in steps)
                self.outliers_cb.setChecked("remove_outliers" in steps)
                self.missing_cb.setChecked("fill_missing" in steps)
                self.smooth_cb.setChecked("smooth" in steps)
                self.baseline_cb.setChecked("baseline_correction" in steps)
                self.peak_cb.setChecked("peak_detection" in steps)

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
        if self.app_data.get("dataset") is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please import data first.")
            return

        # Get current experiment ID from app_data
        current_experiment_id = self.app_data.get("current_experiment_id")
        if current_experiment_id is None:
            QMessageBox.warning(self, "Warning", "No active experiment. Please create or load an experiment first.")
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
                current_experiment_id,
                self.app_data["processed_data"],
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
        if "processed_data" in self.app_data:
            self.app_data["processed_data"] = None
            self.plot_data("raw")
            QMessageBox.information(self, "Reset", "Reset to original data")
