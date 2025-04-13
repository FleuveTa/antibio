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

# Check if the tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features';")
features_exists = cursor.fetchone() is not None

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='labels';")
labels_exists = cursor.fetchone() is not None

print(f"Features table exists: {features_exists}")
print(f"Labels table exists: {labels_exists}")

# Drop the unused tables
if features_exists:
    cursor.execute("DROP TABLE features")
    print("Dropped features table")

if labels_exists:
    cursor.execute("DROP TABLE labels")
    print("Dropped labels table")

# Commit changes
conn.commit()

# Verify tables were dropped
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('Tables in database after dropping:')
for table in tables:
    print(f'- {table[0]}')

# Close the connection
conn.close()
