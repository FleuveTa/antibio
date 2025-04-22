import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, detrend
from scipy.stats import zscore
from sklearn.impute import SimpleImputer


def preprocess_data(df, data_type=None, options=None):
    """
    Preprocess voltammetric data with the specified options.
    Focus on data cleaning, handling missing values, and noise reduction.

    Args:
        df: DataFrame with voltammetry data
        data_type: Type of data (e.g., "Voltammetric") - kept for backward compatibility
        options: Dictionary of preprocessing options

    Returns:
        Processed DataFrame and a dictionary of transformers
    """
    processed_df = df.copy()
    transformers = {}

    # Preserve metadata columns
    metadata_columns = [col for col in processed_df.columns
                       if col not in ["Potential", "Current", "RowIndex"]]

    # Fill missing values
    if options.get("fill_missing", False):
        # Only apply to Potential and Current columns
        numeric_cols = ["Potential", "Current"]

        # Create a copy of these columns for imputation
        numeric_data = processed_df[numeric_cols].copy()

        # Apply imputation
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numeric_data)

        # Replace the original columns with imputed data
        processed_df[numeric_cols] = imputed_data

        # Store the transformer
        transformers["impute"] = imputer

        print(f"Filled {imputer.statistics_.sum()} missing values")

    # Remove outliers
    if options.get("remove_outliers", False):
        # Only consider Potential and Current for outlier detection
        numeric_cols = ["Potential", "Current"]

        # Calculate z-scores
        z_scores = np.abs(zscore(processed_df[numeric_cols], nan_policy='omit'))

        # Create a mask for rows to keep (z-score < 3)
        mask = (z_scores < 3).all(axis=1)

        # Count outliers
        outlier_count = (~mask).sum()

        # Apply the mask to filter out outliers
        if outlier_count > 0:
            processed_df = processed_df[mask].reset_index(drop=True)
            print(f"Removed {outlier_count} outliers")
        else:
            print("No outliers detected")

    # Signal smoothing
    if options.get("smooth", False):
        # Ensure we have enough data points for smoothing
        if len(processed_df) >= 5:
            # Calculate appropriate window length (must be odd and less than data length)
            window_length = min(51, len(processed_df) // 2 * 2 - 1)

            # Ensure window length is at least 5 (minimum for Savitzky-Golay filter)
            window_length = max(5, window_length)

            # Apply Savitzky-Golay filter to Current column
            processed_df["Current"] = savgol_filter(
                processed_df["Current"].values,
                window_length,
                3  # Polynomial order
            )

            print(f"Applied smoothing with window length {window_length}")
        else:
            print("Not enough data points for smoothing")

    # Baseline correction
    if options.get("baseline_correction", False):
        # Apply detrending to remove linear trend from Current
        processed_df["Current"] = detrend(processed_df["Current"].values)
        print("Applied baseline correction")

    # Normalization has been removed as it's not necessary for electrochemical data
    # and can affect the electrochemical relationships

    # Peak detection
    if options.get("detect_peaks", True):  # Enable by default
        try:
            # Ensure data is sorted by Potential for proper peak detection
            processed_df = processed_df.sort_values(by="Potential").reset_index(drop=True)

            # Find peaks in the Current signal
            # Adjust parameters based on your specific data characteristics
            peaks, _ = find_peaks(processed_df["Current"], height=0.1*processed_df["Current"].max(),
                                 distance=5, prominence=0.2*processed_df["Current"].std())

            # Create a Peaks column (boolean)
            processed_df["Peaks"] = False
            if len(peaks) > 0:
                processed_df.loc[peaks, "Peaks"] = True
                print(f"Detected {len(peaks)} peaks in the voltammetry data")
            else:
                print("No peaks detected in the voltammetry data")

            # Also find negative peaks (valleys)
            valleys, _ = find_peaks(-processed_df["Current"], height=0.1*(-processed_df["Current"]).max(),
                                   distance=5, prominence=0.2*processed_df["Current"].std())

            # Add valleys to the Peaks column
            if len(valleys) > 0:
                processed_df.loc[valleys, "Peaks"] = True
                print(f"Detected {len(valleys)} valleys in the voltammetry data")
            else:
                print("No valleys detected in the voltammetry data")

        except Exception as e:
            print(f"Error during peak detection: {e}")

    return processed_df, transformers


def apply_transformers(df, transformers):
    """
    Apply saved preprocessing transformers to new data.

    Args:
        df: DataFrame to transform
        transformers: Dictionary of transformers

    Returns:
        Transformed DataFrame
    """
    processed_df = df.copy()

    # Apply transformers in a specific order
    transformer_order = ["impute", "normalize"]

    for step_name in transformer_order:
        if step_name in transformers:
            transformer = transformers[step_name]

            if step_name in ["impute", "normalize"]:
                # These transformers work on specific columns
                numeric_cols = ["Potential", "Current"]

                # Apply transformer
                transformed_data = transformer.transform(processed_df[numeric_cols])

                # Replace the original columns with transformed data
                processed_df[numeric_cols] = transformed_data

    return processed_df
