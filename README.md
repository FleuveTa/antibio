# ElectroSense - Antibiotic Residue Detection

An application for analyzing voltammetric data to detect antibiotic residues.

## Data Handling Guidelines

### Data Files

This project uses various data files for analysis, but these files are not tracked in the Git repository due to their size and potentially sensitive nature. The `.gitignore` file is configured to exclude the following file types:

- CSV files (*.csv)
- Excel files (*.xlsx, *.xls)
- Text files (*.txt)
- Data files (*.dat)
- JSON files (*.json)
- Database files (*.db, *.sqlite, *.sqlite3)

### Working with Data Files

When working with this project, you'll need to:

1. **Store data files locally**: Place your data files in the `data/` directory.
2. **Never commit data files**: Avoid using `git add` on data files or the data directory.
3. **Share data separately**: If you need to share data with collaborators, use alternative methods like cloud storage or secure file transfer.

### Data Directory Structure

The application expects the following data directory structure:

```
data/
├── raw/              # Raw data files
├── processed/        # Processed data
├── features/         # Extracted features
├── models/           # Trained models
└── transformers/     # Preprocessing transformers
```

This structure is automatically created when you run the application for the first time.

### Ignoring Specific Rows in Data Files

When importing data, you can specify rows to ignore using the "Skip Rows" field in the Data Import tab:

- Enter comma-separated row indices (e.g., `0,1,2`)
- Enter a range of rows (e.g., `0-5`)
- Combine both formats (e.g., `0,2,5-10`)

Note that row indices start at 0 (the first row is row 0).

## Installation and Setup

[Add installation instructions here]

## Usage

[Add usage instructions here]

## License

[Add license information here]
