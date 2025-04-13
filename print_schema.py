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
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = cursor.fetchall()
print(f"Found {len(tables)} tables in the database")
print("=" * 80)

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
    
    # Print table header
    print(f"TABLE: {table_name}")
    print("-" * 80)
    
    # Print columns
    print(f"{'Column':<20} {'Type':<15} {'PK':<5} {'NN':<5} {'Default':<20}")
    print("-" * 80)
    
    for col in columns:
        col_id = col[0]
        col_name = col[1]
        col_type = col[2]
        not_null = "Yes" if col[3] == 1 else "No"
        default_val = str(col[4]) if col[4] is not None else ""
        is_pk = "Yes" if col[5] == 1 else "No"
        
        print(f"{col_name:<20} {col_type:<15} {is_pk:<5} {not_null:<5} {default_val:<20}")
    
    print()

# Print relationships
print("\nRELATIONSHIPS")
print("=" * 80)

for table_name, fks in foreign_keys.items():
    for fk in fks:
        print(f"{table_name}.{fk['from_col']} -> {fk['to_table']}.{fk['to_col']}")

# Get row counts
print("\nROW COUNTS")
print("=" * 80)

for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"{table_name}: {count} rows")

# Close the connection
conn.close()
