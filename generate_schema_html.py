import sqlite3
import os
import html

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

# Create HTML file
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Database Schema</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .table {
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            width: 300px;
        }
        .table-header {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            font-weight: bold;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        .table-content {
            padding: 10px;
        }
        .column {
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
        .column:last-child {
            border-bottom: none;
        }
        .primary-key {
            font-weight: bold;
            color: #FF5722;
        }
        .foreign-key {
            color: #2196F3;
            font-style: italic;
        }
        .not-null {
            text-decoration: underline;
        }
        .relationship {
            margin: 20px 0;
            padding: 10px;
            background-color: #e1f5fe;
            border-radius: 5px;
            border-left: 5px solid #2196F3;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .legend {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .legend div {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <h1>Database Schema</h1>
    <div class="container">
"""

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

    # Create table HTML
    html_content += f"""
        <div class="table">
            <div class="table-header">{html.escape(table_name)}</div>
            <div class="table-content">
    """

    # Add columns
    for col in columns:
        col_id = col[0]
        col_name = col[1]
        col_type = col[2]
        not_null = col[3] == 1
        default_val = col[4]
        is_pk = col[5] == 1

        # Check if column is a foreign key
        is_fk = False
        for fk_table, fk_list in foreign_keys.items():
            if fk_table == table_name:
                for fk in fk_list:
                    if fk['from_col'] == col_name:
                        is_fk = True

        # Apply appropriate CSS classes
        classes = []
        if is_pk:
            classes.append("primary-key")
        if is_fk:
            classes.append("foreign-key")
        if not_null:
            classes.append("not-null")

        class_str = " ".join(classes)
        class_attr = f' class="{class_str}"' if class_str else ""

        # Add column details
        html_content += f"""
                <div class="column"{class_attr}>
                    {html.escape(col_name)} : {html.escape(col_type)}
                    {" (PK)" if is_pk else ""}
                    {" NOT NULL" if not_null else ""}
                    {f" DEFAULT {html.escape(str(default_val))}" if default_val is not None else ""}
                </div>
        """

    html_content += """
            </div>
        </div>
    """

# Add relationships section
html_content += """
    </div>

    <h2>Relationships</h2>
"""

for table_name, fks in foreign_keys.items():
    for fk in fks:
        html_content += f"""
    <div class="relationship">
        {html.escape(table_name)}.{html.escape(fk['from_col'])} -&gt; {html.escape(fk['to_table'])}.{html.escape(fk['to_col'])}
    </div>
        """

# Add legend
html_content += """
    <div class="legend">
        <h3>Legend</h3>
        <div class="primary-key">Primary Key</div>
        <div class="foreign-key">Foreign Key</div>
        <div class="not-null">NOT NULL constraint</div>
    </div>
</body>
</html>
"""

# Write HTML to file
output_file = 'database_schema.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Schema HTML saved as {output_file}")

# Close the connection
conn.close()
