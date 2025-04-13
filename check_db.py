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

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('Tables in database:')
for table in tables:
    print(f'- {table[0]}')

# For each table, get schema and count rows
print('\nTable details:')
for table in tables:
    table_name = table[0]
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    
    print(f"\n{table_name} ({row_count} rows):")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

# Check for foreign key relationships
print('\nForeign key relationships:')
for table in tables:
    table_name = table[0]
    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
    foreign_keys = cursor.fetchall()
    
    if foreign_keys:
        print(f"\n{table_name} references:")
        for fk in foreign_keys:
            print(f"  - {fk[2]}({fk[3]}) -> {fk[0]}({fk[2]})")

# Close the connection
conn.close()
