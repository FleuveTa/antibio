import sqlite3
import os
from graphviz import Digraph

# Find the database file
db_path = 'data/experiments.db'
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = cursor.fetchall()
print(f"Found {len(tables)} tables in the database")

# Create a new graph
dot = Digraph(comment='Database Schema', format='png')
dot.attr('graph', rankdir='LR', ratio='fill', size='8,4')
dot.attr('node', shape='record', style='filled', fillcolor='lightblue')

# Foreign key relationships
foreign_keys = {}

# Process each table
for table in tables:
    table_name = table[0]
    
    # Get table columns
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    # Get foreign keys
    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
    fks = cursor.fetchall()
    
    # Store foreign key relationships
    for fk in fks:
        if table_name not in foreign_keys:
            foreign_keys[table_name] = []
        foreign_keys[table_name].append({
            'from_col': fk[3],
            'to_table': fk[2],
            'to_col': fk[4]
        })
    
    # Create table node
    table_label = f"{{{table_name}|"
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        pk = " (PK)" if col[5] == 1 else ""
        nn = " NOT NULL" if col[3] == 1 else ""
        table_label += f"{col_name} : {col_type}{pk}{nn}\\l"
    table_label += "}"
    
    dot.node(table_name, table_label)

# Add edges for foreign key relationships
for table_name, fks in foreign_keys.items():
    for fk in fks:
        dot.edge(f"{table_name}:{fk['from_col']}", f"{fk['to_table']}:{fk['to_col']}")

# Render the graph
output_file = 'database_schema'
dot.render(output_file, view=True)
print(f"Schema diagram saved as {output_file}.png")

# Close the connection
conn.close()
