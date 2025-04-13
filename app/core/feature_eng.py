import numpy as np
import pandas as pd
from scipy import signal, integrate
import re


def extract_voltammetric_features(df):
    """
    Extract features from voltammetric data, organized into categories
    that are meaningful for electrochemical analysis.

    Returns a dictionary with feature categories as keys and feature dictionaries as values.
    """
    # Initialize feature categories
    features = {
        "basic": {},       # Basic statistical features
        "peak": {},        # Peak-related features
        "shape": {},       # Shape-related features
        "derivative": {},  # Derivative-based features
        "area": {},        # Area-related features
    }

    # ---- Basic Statistical Features ----
    features["basic"]["mean_current"] = df["Current"].mean()
    features["basic"]["median_current"] = df["Current"].median()
    features["basic"]["std_current"] = df["Current"].std()
    features["basic"]["min_current"] = df["Current"].min()
    features["basic"]["max_current"] = df["Current"].max()
    features["basic"]["current_range"] = df["Current"].max() - df["Current"].min()

    # ---- Peak Features ----
    if "Peaks" in df.columns:
        peak_indices = df[df["Peaks"]].index
        if len(peak_indices) > 0:
            peak_currents = df.loc[peak_indices, "Current"]
            peak_potentials = df.loc[peak_indices, "Potential"]

            # Number of peaks
            features["peak"]["num_peaks"] = len(peak_indices)

            # Maximum peak features
            max_peak_idx = peak_currents.argmax()
            features["peak"]["max_peak_current"] = peak_currents.max()
            features["peak"]["max_peak_potential"] = peak_potentials.iloc[max_peak_idx]

            # Minimum peak features
            min_peak_idx = peak_currents.argmin()
            features["peak"]["min_peak_current"] = peak_currents.min()
            features["peak"]["min_peak_potential"] = peak_potentials.iloc[min_peak_idx]

            # Peak-to-peak distances
            if len(peak_indices) >= 2:
                peak_distances = np.diff(peak_potentials)
                features["peak"]["mean_peak_distance"] = peak_distances.mean()
                features["peak"]["max_peak_distance"] = peak_distances.max()
                features["peak"]["min_peak_distance"] = peak_distances.min()

                # Peak height ratio (if there are at least 2 peaks)
                sorted_peaks = sorted(peak_currents)
                if len(sorted_peaks) >= 2:
                    features["peak"]["peak_ratio"] = sorted_peaks[-1] / sorted_peaks[-2] if sorted_peaks[-2] != 0 else 0

    # ---- Shape Features ----
    # Skewness and kurtosis of the current distribution
    features["shape"]["current_skewness"] = df["Current"].skew()
    features["shape"]["current_kurtosis"] = df["Current"].kurtosis()

    # Symmetry measure (ratio of area before and after the max current point)
    max_current_idx = df["Current"].idxmax()
    before_max = df.iloc[:max_current_idx+1]
    after_max = df.iloc[max_current_idx:]

    if not before_max.empty and not after_max.empty:
        area_before = integrate.trapz(before_max["Current"], before_max["Potential"])
        area_after = integrate.trapz(after_max["Current"], after_max["Potential"])
        if area_after != 0:
            features["shape"]["symmetry_ratio"] = abs(area_before / area_after) if area_after != 0 else 0

    # ---- Derivative Features ----
    # Calculate first derivative of current with respect to potential
    if len(df) > 1:
        df_sorted = df.sort_values(by="Potential")
        current_gradient = np.gradient(df_sorted["Current"].values, df_sorted["Potential"].values)

        features["derivative"]["max_gradient"] = np.max(current_gradient)
        features["derivative"]["min_gradient"] = np.min(current_gradient)
        features["derivative"]["mean_gradient"] = np.mean(current_gradient)
        features["derivative"]["std_gradient"] = np.std(current_gradient)

        # Second derivative features
        second_gradient = np.gradient(current_gradient, df_sorted["Potential"].values)
        features["derivative"]["max_second_gradient"] = np.max(second_gradient)
        features["derivative"]["min_second_gradient"] = np.min(second_gradient)

    # ---- Area Features ----
    # Total area under the curve (integral of current vs. potential)
    if "Potential" in df.columns and "Current" in df.columns:
        # Sort by potential for proper integration
        df_sorted = df.sort_values(by="Potential")
        area = integrate.trapz(df_sorted["Current"], df_sorted["Potential"])
        features["area"]["total_area"] = area

        # Positive and negative areas
        positive_df = df_sorted[df_sorted["Current"] > 0]
        negative_df = df_sorted[df_sorted["Current"] < 0]

        if not positive_df.empty:
            pos_area = integrate.trapz(positive_df["Current"], positive_df["Potential"])
            features["area"]["positive_area"] = pos_area
        else:
            features["area"]["positive_area"] = 0

        if not negative_df.empty:
            neg_area = integrate.trapz(negative_df["Current"], negative_df["Potential"])
            features["area"]["negative_area"] = neg_area
        else:
            features["area"]["negative_area"] = 0

        # Area ratio
        if features["area"]["negative_area"] != 0:
            features["area"]["area_ratio"] = features["area"]["positive_area"] / abs(features["area"]["negative_area"]) \
                if features["area"]["negative_area"] != 0 else 0

    # Flatten the nested dictionary for easier use in the UI
    flat_features = {}
    for category, category_features in features.items():
        for feature_name, feature_value in category_features.items():
            flat_features[f"{category}_{feature_name}"] = feature_value

    return flat_features


def extract_time_series_features(df, target_column):
    """
    Extract features from time series data
    """
    features = {}
    series = df[target_column]

    # Time-domain features
    features["mean"] = series.mean()
    features["std"] = series.std()
    features["min"] = series.min()
    features["max"] = series.max()
    features["range"] = series.max() - series.min()
    features["skewness"] = series.skew()
    features["kurtosis"] = series.kurtosis()

    # First and second derivatives
    features["mean_gradient"] = np.gradient(series).mean()
    features["std_gradient"] = np.gradient(series).std()
    features["mean_second_derivative"] = np.gradient(np.gradient(series)).mean()

    # Frequency domain features (FFT)
    fft = np.fft.fft(series)
    fft_magnitude = np.abs(fft)
    features["fft_mean"] = fft_magnitude.mean()
    features["fft_std"] = fft_magnitude.std()
    features["fft_max"] = fft_magnitude.max()

    return features


def extract_features_from_samples(df):
    """
    Extract features from voltammetric data on a sample-by-sample basis.

    This function processes a dataset where:
    - Each row represents a sample
    - Columns include voltage points (e.g., -0.8, -0.79, etc.)
    - Additional metadata columns like 'concentration' and 'antibiotic' are present

    Args:
        df: DataFrame with voltage columns and metadata columns

    Returns:
        DataFrame with extracted features for each sample and preserved metadata
    """
    # Identify voltage columns and metadata columns
    voltage_columns = []
    metadata_columns = []

    for col in df.columns:
        # Check if column name can be converted to a float (voltage point)
        try:
            # Try direct conversion
            float(col)
            voltage_columns.append(col)
        except ValueError:
            # Check for complex voltage format like "-0.795.1925"
            if isinstance(col, str) and re.match(r'^-?\d+\.\d+(\.\d+)*$', col):
                # Extract the first part as the voltage value
                parts = col.split('.')
                if len(parts) >= 2:
                    try:
                        float(f"{parts[0]}.{parts[1]}")
                        voltage_columns.append(col)
                    except ValueError:
                        metadata_columns.append(col)
            # If not a voltage column, it's a metadata column
            elif col not in ['path']:  # Skip 'path' column
                metadata_columns.append(col)

    # Create a list to store features for each sample
    all_features = []

    # Process each sample (row) individually
    for idx, row in df.iterrows():
        # Extract current values at each voltage point
        voltammetry_data = []

        for col in voltage_columns:
            try:
                # Get voltage value
                if '.' in col:
                    parts = col.split('.')
                    voltage = float(f"{parts[0]}.{parts[1]}")
                else:
                    voltage = float(col)

                # Get current value
                current = float(row[col])

                # Add to voltammetry data
                voltammetry_data.append({'Potential': voltage, 'Current': current})
            except (ValueError, TypeError):
                # Skip if conversion fails
                continue

        # Convert to DataFrame for feature extraction
        if voltammetry_data:
            voltammetry_df = pd.DataFrame(voltammetry_data)

            # Sort by potential
            voltammetry_df = voltammetry_df.sort_values(by='Potential')

            # Extract features for this sample
            sample_features = extract_voltammetric_features(voltammetry_df)

            # Add metadata
            for col in metadata_columns:
                if col in row.index:
                    sample_features[col] = row[col]

            # Add to the list of all features
            all_features.append(sample_features)

    # Convert to DataFrame
    if all_features:
        features_df = pd.DataFrame(all_features)
        return features_df
    else:
        return pd.DataFrame()


# Dictionary of feature descriptions for user-friendly display
VOLTAMMETRIC_FEATURE_DESCRIPTIONS = {
    # Basic features
    "basic_mean_current": "Average current value across the entire voltammogram",
    "basic_median_current": "Middle current value when all measurements are arranged in order",
    "basic_std_current": "Standard deviation of current, indicating variability",
    "basic_min_current": "Minimum current value in the voltammogram",
    "basic_max_current": "Maximum current value in the voltammogram",
    "basic_current_range": "Difference between maximum and minimum current",

    # Peak features
    "peak_num_peaks": "Number of peaks detected in the voltammogram",
    "peak_max_peak_current": "Current value at the highest peak",
    "peak_max_peak_potential": "Potential value at the highest peak",
    "peak_min_peak_current": "Current value at the lowest peak",
    "peak_min_peak_potential": "Potential value at the lowest peak",
    "peak_mean_peak_distance": "Average distance between adjacent peaks",
    "peak_max_peak_distance": "Maximum distance between adjacent peaks",
    "peak_min_peak_distance": "Minimum distance between adjacent peaks",
    "peak_peak_ratio": "Ratio of the highest peak to the second highest peak",

    # Shape features
    "shape_current_skewness": "Asymmetry of the current distribution",
    "shape_current_kurtosis": "Peakedness of the current distribution",
    "shape_symmetry_ratio": "Ratio of area before and after the maximum current point",

    # Derivative features
    "derivative_max_gradient": "Maximum rate of change of current with respect to potential",
    "derivative_min_gradient": "Minimum rate of change of current with respect to potential",
    "derivative_mean_gradient": "Average rate of change of current with respect to potential",
    "derivative_std_gradient": "Variability in the rate of change of current",
    "derivative_max_second_gradient": "Maximum acceleration of current change",
    "derivative_min_second_gradient": "Minimum acceleration of current change",

    # Area features
    "area_total_area": "Total area under the current-potential curve",
    "area_positive_area": "Area under the curve where current is positive",
    "area_negative_area": "Area under the curve where current is negative",
    "area_area_ratio": "Ratio of positive area to negative area"
}

# Dictionary of feature categories with descriptions
FEATURE_CATEGORIES = {
    "basic": "Basic statistical features of the current signal",
    "peak": "Features related to peaks in the voltammogram",
    "shape": "Features describing the shape of the voltammogram",
    "derivative": "Features based on the rate of change of current",
    "area": "Features related to the area under the voltammogram"
}


def select_features(feature_df, target=None, method="importance", threshold=0.2):
    """
    Select relevant features based on different methods
    """
    if target is None or feature_df.empty:
        # If no target or empty dataframe, return all features
        return feature_df.columns.tolist()

    if method == "correlation" and target is not None:
        # Calculate correlation with target
        correlations = feature_df.corrwith(target).abs()
        selected_features = correlations[correlations > threshold].index.tolist()

    elif method == "variance":
        # Remove low-variance features
        variances = feature_df.var()
        selected_features = variances[variances > threshold].index.tolist()

    elif method == "importance":
        # Use a simple feature importance method (Random Forest)
        from sklearn.ensemble import RandomForestRegressor

        # Handle case where target is a Series
        if hasattr(target, 'values'):
            target_values = target.values
        else:
            target_values = target

        # Fill NaN values
        feature_df_filled = feature_df.fillna(0)

        # Train a Random Forest model
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(feature_df_filled, target_values)

        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Select top features
        num_features = max(1, int(len(feature_df.columns) * threshold))
        selected_features = [feature_df.columns[i] for i in indices[:num_features]]

    else:
        # Default: return all features
        selected_features = feature_df.columns.tolist()

    return selected_features