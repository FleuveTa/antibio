from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QSplitter, QTextEdit, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
import sqlite3
import os
from pathlib import Path

class DatabaseViewer(QWidget):
    """
    A simple database viewer widget for examining SQLite database structure
    """
    def __init__(self, db_path=None):
        super().__init__()
        self.db_path = db_path or os.path.join("data", "experiments.db")
        self.conn = None
        self.cursor = None
        self.init_ui()
        self.connect_db()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        self.setWindowTitle("Database Structure Viewer")
        self.resize(800, 600)
        
        # Database connection info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Database:"))
        self.db_label = QLabel(self.db_path)
        self.db_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.db_label)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_view)
        info_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(info_layout)
        
        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # Tables list
        tables_group = QGroupBox("Database Tables")
        tables_layout = QVBoxLayout()
        
        self.tables_combo = QComboBox()
        self.tables_combo.currentIndexChanged.connect(self.table_selected)
        tables_layout.addWidget(self.tables_combo)
        
        tables_group.setLayout(tables_layout)
        splitter.addWidget(tables_group)
        
        # Table structure
        structure_group = QGroupBox("Table Structure")
        structure_layout = QVBoxLayout()
        
        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(4)
        self.structure_table.setHorizontalHeaderLabels(["Column", "Type", "Not Null", "Default"])
        self.structure_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        structure_layout.addWidget(self.structure_table)
        
        structure_group.setLayout(structure_layout)
        splitter.addWidget(structure_group)
        
        # Table data
        data_group = QGroupBox("Table Data")
        data_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        data_layout.addWidget(self.data_table)
        
        data_group.setLayout(data_layout)
        splitter.addWidget(data_group)
        
        # SQL query section
        query_group = QGroupBox("SQL Query")
        query_layout = QVBoxLayout()
        
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Enter SQL query here...")
        query_layout.addWidget(self.query_edit)
        
        query_btn_layout = QHBoxLayout()
        self.run_query_btn = QPushButton("Run Query")
        self.run_query_btn.clicked.connect(self.run_query)
        query_btn_layout.addWidget(self.run_query_btn)
        
        self.schema_btn = QPushButton("Show Schema")
        self.schema_btn.clicked.connect(self.show_schema)
        query_btn_layout.addWidget(self.schema_btn)
        
        query_layout.addLayout(query_btn_layout)
        
        query_group.setLayout(query_layout)
        splitter.addWidget(query_group)
        
        # Set initial sizes
        splitter.setSizes([100, 200, 300, 200])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
    def connect_db(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.load_tables()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error connecting to database: {str(e)}")
    
    def load_tables(self):
        """Load the list of tables from the database"""
        try:
            self.tables_combo.clear()
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            for table in tables:
                self.tables_combo.addItem(table[0])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading tables: {str(e)}")
    
    def table_selected(self, index):
        """Handle table selection from combo box"""
        if index < 0:
            return
        
        table_name = self.tables_combo.itemText(index)
        self.load_table_structure(table_name)
        self.load_table_data(table_name)
        
        # Set a default query for the selected table
        self.query_edit.setText(f"SELECT * FROM {table_name} LIMIT 100;")
    
    def load_table_structure(self, table_name):
        """Load the structure of the selected table"""
        try:
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            
            self.structure_table.setRowCount(len(columns))
            for i, col in enumerate(columns):
                # col format: (cid, name, type, notnull, dflt_value, pk)
                self.structure_table.setItem(i, 0, QTableWidgetItem(col[1]))  # name
                self.structure_table.setItem(i, 1, QTableWidgetItem(col[2]))  # type
                self.structure_table.setItem(i, 2, QTableWidgetItem("Yes" if col[3] else "No"))  # notnull
                self.structure_table.setItem(i, 3, QTableWidgetItem(str(col[4]) if col[4] is not None else ""))  # default
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading table structure: {str(e)}")
    
    def load_table_data(self, table_name):
        """Load the data from the selected table"""
        try:
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 100;")
            data = self.cursor.fetchall()
            
            # Get column names
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            self.data_table.setColumnCount(len(columns))
            self.data_table.setHorizontalHeaderLabels(columns)
            
            self.data_table.setRowCount(len(data))
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    self.data_table.setItem(i, j, QTableWidgetItem(str(value)))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading table data: {str(e)}")
    
    def run_query(self):
        """Run the SQL query entered by the user"""
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        
        try:
            self.cursor.execute(query)
            
            # Check if the query returns data
            if self.cursor.description:
                data = self.cursor.fetchall()
                columns = [col[0] for col in self.cursor.description]
                
                self.data_table.setColumnCount(len(columns))
                self.data_table.setHorizontalHeaderLabels(columns)
                
                self.data_table.setRowCount(len(data))
                for i, row in enumerate(data):
                    for j, value in enumerate(row):
                        self.data_table.setItem(i, j, QTableWidgetItem(str(value)))
                
                QMessageBox.information(self, "Success", f"Query executed successfully. {len(data)} rows returned.")
            else:
                # For non-SELECT queries
                self.conn.commit()
                QMessageBox.information(self, "Success", "Query executed successfully.")
                self.refresh_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error executing query: {str(e)}")
    
    def show_schema(self):
        """Show the database schema"""
        try:
            self.cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
            schema = self.cursor.fetchall()
            
            schema_text = "\n\n".join([row[0] for row in schema if row[0]])
            
            self.query_edit.setText(schema_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error retrieving schema: {str(e)}")
    
    def refresh_view(self):
        """Refresh the database view"""
        if self.conn:
            self.conn.close()
        self.connect_db()
        
        # Reload the current table if one is selected
        current_index = self.tables_combo.currentIndex()
        if current_index >= 0:
            self.table_selected(current_index)
    
    def closeEvent(self, event):
        """Clean up when the widget is closed"""
        if self.conn:
            self.conn.close()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Try to find the database file
    db_path = None
    possible_paths = [
        os.path.join("data", "experiments.db"),
        os.path.join("..", "data", "experiments.db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "experiments.db")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    viewer = DatabaseViewer(db_path)
    viewer.show()
    
    sys.exit(app.exec_())
