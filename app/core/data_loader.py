import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.signal import find_peaks
from scipy.signal import detrend


def load_data(file_path):
    """
    Load voltammetric data from a file.
    Supports CSV files with both comma and semicolon delimiters.
    Handles voltammetry data where columns are voltage values and rows contain current measurements.
    """
    try:
        # First, try to detect if this is a standard format or the special voltammetry format
        # Try loading with comma separator first
        try:
            df = pd.read_csv(file_path)
        except:
            # Try with semicolon separator
            df = pd.read_csv(file_path, sep=';')

        # Check if this is the special voltammetry format (columns are voltage values)
        # In this format, the first row contains voltage values as column headers
        is_voltammetry_format = False

        # Check if column headers can be converted to float (indicating they are voltage values)
        voltage_columns = []
        for col in df.columns[1:]:  # Skip first column which might be an index
            try:
                float(col)
                voltage_columns.append(col)
                is_voltammetry_format = True
            except (ValueError, TypeError):
                pass

        if is_voltammetry_format and len(voltage_columns) > 0:
            print(f"Detected voltammetry format with {len(voltage_columns)} voltage columns")

            # Convert the data to the format we need (Potential and Current columns)
            # First, convert column headers to float
            voltage_values = [float(col) for col in voltage_columns]

            # Create a new dataframe with Potential and Current columns
            new_data = []

            # For each row in the original dataframe
            for _, row in df.iterrows():
                # For each voltage column
                for i, voltage_col in enumerate(voltage_columns):
                    # Add a row with Potential = voltage value, Current = cell value
                    try:
                        current_value = float(row[voltage_col])
                        new_data.append({
                            'Potential': float(voltage_col),
                            'Current': current_value
                        })
                    except (ValueError, TypeError):
                        # Skip cells that can't be converted to float
                        pass

            # Create new dataframe
            new_df = pd.DataFrame(new_data)
            return new_df

        # If not the special format, check if we have the required columns
        if "Potential" not in df.columns or "Current" not in df.columns:
            # Try to find columns with units in parentheses
            potential_col = next((col for col in df.columns if "Potential" in str(col)), None)
            current_col = next((col for col in df.columns if "Current" in str(col)), None)

            if potential_col and current_col:
                df = df.rename(columns={
                    potential_col: "Potential",
                    current_col: "Current"
                })
            else:
                raise ValueError("Could not find 'Potential' and 'Current' columns in the data file")

        # Convert string values to numeric if needed
        df["Potential"] = pd.to_numeric(df["Potential"], errors='coerce')
        df["Current"] = pd.to_numeric(df["Current"], errors='coerce')

        return df

    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def preprocess_data(df, sensor_type, options):
    """
    Preprocess voltammetric data with the specified options.
    Returns the processed DataFrame and a dictionary of transformers.
    """
    processed_df = df.copy()
    transformers = {}

    # Fill missing values
    if options.get("fill_missing", False):
        imputer = SimpleImputer(strategy='mean')
        processed_df[["Potential", "Current"]] = imputer.fit_transform(processed_df[["Potential", "Current"]])
        transformers["impute"] = imputer

    # Normalize data
    if options.get("normalize", False):
        scaler = StandardScaler()
        processed_df[["Potential", "Current"]] = scaler.fit_transform(processed_df[["Potential", "Current"]])
        transformers["normalize"] = scaler

    # Remove outliers
    if options.get("remove_outliers", False):
        z_scores = np.abs(zscore(processed_df[["Potential", "Current"]]))
        processed_df = processed_df[(z_scores < 3).all(axis=1)]

    # Signal smoothing
    if options.get("smooth", False):
        window_length = min(51, len(processed_df) // 2 * 2 - 1)  # Ensure odd window length
        processed_df["Current"] = savgol_filter(processed_df["Current"], window_length, 3)

    # Baseline correction
    if options.get("baseline_correction", False):
        processed_df["Current"] = detrend(processed_df["Current"])

    # Peak detection
    if options.get("peak_detection", False):
        peaks, _ = find_peaks(processed_df["Current"], height=0)
        processed_df["Peaks"] = False
        processed_df.loc[peaks, "Peaks"] = True

    return processed_df, transformers