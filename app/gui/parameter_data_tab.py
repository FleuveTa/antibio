from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QFileDialog, QTableWidget,
                             QTableWidgetItem, QGroupBox, QGridLayout, QLineEdit,
                             QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ParameterDataImportTab(QWidget):
    """
    Tab for importing parameter data for optimization
    """
    data_loaded = pyqtSignal(bool)
    
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File Import Section
        import_group = QGroupBox("Import Parameter Data")
        import_layout = QGridLayout()
        
        # File selection
        import_layout.addWidget(QLabel("Data File:"), 0, 0)
        self.file_path = QLineEdit("No file selected")
        self.file_path.setReadOnly(True)
        import_layout.addWidget(self.file_path, 0, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_files)
        import_layout.addWidget(self.browse_btn, 0, 2)
        
        # Parameter column selection
        import_layout.addWidget(QLabel("Parameter Column:"), 1, 0)
        self.param_column = QComboBox()
        import_layout.addWidget(self.param_column, 1, 1, 1, 2)
        
        # Response column selection
        import_layout.addWidget(QLabel("Peak Current Column:"), 2, 0)
        self.response_column = QComboBox()
        import_layout.addWidget(self.response_column, 2, 1, 1, 2)
        
        # Import button
        self.import_btn = QPushButton("Import Data")
        self.import_btn.clicked.connect(self.import_data)
        import_layout.addWidget(self.import_btn, 3, 0, 1, 3)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Data Preview Section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        # Table for data preview
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(["Parameter", "Peak Current"])
        preview_layout.addWidget(self.data_table)
        
        # Plot for data visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        preview_layout.addWidget(self.canvas)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.setLayout(layout)
    
    def browse_files(self):
        """Open file dialog to select data file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self.load_columns(file_path)
    
    def load_columns(self, file_path):
        """Load column names from the selected file"""
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                # Try with different delimiters
                try:
                    df = pd.read_csv(file_path)
                except:
                    df = pd.read_csv(file_path, sep=';')
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Store the dataframe temporarily
            self.temp_df = df
            
            # Update column selection dropdowns
            self.param_column.clear()
            self.response_column.clear()
            
            # Add numeric columns to the dropdowns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.param_column.addItems(numeric_cols)
            self.response_column.addItems(numeric_cols)
            
            # Set default selections if possible
            if len(numeric_cols) >= 2:
                self.param_column.setCurrentIndex(0)
                self.response_column.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def import_data(self):
        """Import the selected data"""
        if not hasattr(self, 'temp_df'):
            QMessageBox.warning(self, "Warning", "Please select a data file first")
            return
        
        param_col = self.param_column.currentText()
        response_col = self.response_column.currentText()
        
        if param_col == response_col:
            QMessageBox.warning(self, "Warning", "Parameter and Response columns must be different")
            return
        
        try:
            # Extract the selected columns
            df = self.temp_df[[param_col, response_col]].copy()
            
            # Remove rows with missing values
            df.dropna(inplace=True)
            
            # Store the data in app_data
            self.app_data["parameter_data"] = {
                "df": df,
                "param_col": param_col,
                "response_col": response_col,
                "param_values": df[param_col].values,
                "response_values": df[response_col].values
            }
            
            # Update the data table
            self.update_data_table(df, param_col, response_col)
            
            # Update the plot
            self.update_plot(df, param_col, response_col)
            
            # Emit signal that data is loaded
            self.data_loaded.emit(True)
            
            QMessageBox.information(self, "Success", "Data imported successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importing data: {str(e)}")
    
    def update_data_table(self, df, param_col, response_col):
        """Update the data table with the imported data"""
        # Clear the table
        self.data_table.setRowCount(0)
        
        # Set the column headers
        self.data_table.setHorizontalHeaderLabels([param_col, response_col])
        
        # Add the data rows
        for i, (_, row) in enumerate(df.iterrows()):
            self.data_table.insertRow(i)
            self.data_table.setItem(i, 0, QTableWidgetItem(str(row[param_col])))
            self.data_table.setItem(i, 1, QTableWidgetItem(str(row[response_col])))
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()
    
    def update_plot(self, df, param_col, response_col):
        """Update the plot with the imported data"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot the data
        ax.scatter(df[param_col], df[response_col], color='blue')
        
        # Set labels and title
        ax.set_xlabel(param_col)
        ax.set_ylabel(response_col)
        ax.set_title(f"{response_col} vs {param_col}")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Update the canvas
        self.canvas.draw()
