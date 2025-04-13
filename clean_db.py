import sqlite3
import os

# Find the database file
db_path = 'data/experiments.db'
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a backup of the database
backup_path = 'data/experiments_backup.db'
backup_conn = sqlite3.connect(backup_path)
conn.backup(backup_conn)
backup_conn.close()
print(f"Created backup at {backup_path}")

# Check if the tables exist and have data
cursor.execute("SELECT COUNT(*) FROM features")
features_count = cursor.fetchone()[0]
print(f"Features table has {features_count} rows")

cursor.execute("SELECT COUNT(*) FROM labels")
labels_count = cursor.fetchone()[0]
print(f"Labels table has {labels_count} rows")

# Drop the unused tables
if features_count == 0:
    cursor.execute("DROP TABLE IF EXISTS features")
    print("Dropped features table")

if labels_count == 0:
    cursor.execute("DROP TABLE IF EXISTS labels")
    print("Dropped labels table")

# Commit changes
conn.commit()

# Update the ExperimentManager class to remove references to these tables
experiment_manager_path = 'app/core/experiment_manager.py'
with open(experiment_manager_path, 'r') as f:
    content = f.read()

# Remove the table creation statements
new_content = content

# Remove the features table creation
features_table_start = "        # Create features table"
features_table_end = "        )'''"
if features_table_start in new_content and features_table_end in new_content:
    start_idx = new_content.find(features_table_start)
    end_idx = new_content.find(features_table_end, start_idx) + len(features_table_end)
    new_content = new_content[:start_idx] + new_content[end_idx:]
    print("Removed features table creation from ExperimentManager")

# Remove the labels table creation
labels_table_start = "        # Create labels table"
labels_table_end = "        )'''"
if labels_table_start in new_content and labels_table_end in new_content:
    start_idx = new_content.find(labels_table_start)
    end_idx = new_content.find(labels_table_end, start_idx) + len(labels_table_end)
    new_content = new_content[:start_idx] + new_content[end_idx:]
    print("Removed labels table creation from ExperimentManager")

# Write the updated content back to the file
with open(experiment_manager_path, 'w') as f:
    f.write(new_content)

# Close the connection
conn.close()
print("Database cleanup completed successfully")
