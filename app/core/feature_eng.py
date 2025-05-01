import numpy as np
import pandas as pd
from scipy import integrate
import re
from sklearn.decomposition import PCA


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
    # Note: e_symmetry_ratio has been removed as it has importance < 0.04
    # max_current_idx = df["Current"].idxmax()
    # before_max = df.iloc[:max_current_idx+1]
    # after_max = df.iloc[max_current_idx:]
    #
    # if not before_max.empty and not after_max.empty:
    #     area_before = integrate.trapz(before_max["Current"], before_max["Potential"])
    #     area_after = integrate.trapz(after_max["Current"], after_max["Potential"])
    #     if area_after != 0:
    #         features["shape"]["symmetry_ratio"] = abs(area_before / area_after) if area_after != 0 else 0

    # ---- Derivative Features ----
    # Derivative features have been removed as they are not needed for electrochemical analysis
    # and can introduce noise in the data
    # Just initialize an empty dictionary to maintain structure
    features["derivative"] = {}

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


def extract_features_from_samples(df, filter_features=True):
    """
    Extract features from voltammetric data on a sample-by-sample basis.

    This function processes a dataset where:
    - Each row represents a sample
    - Columns include voltage points (e.g., -0.8, -0.79, etc.)
    - Additional metadata columns like 'concentration' and 'antibiotic' are present

    Args:
        df: DataFrame with voltage columns and metadata columns
        filter_features: Whether to filter out features with errors or low importance

    Returns:
        DataFrame with extracted features for each sample and preserved metadata
    """
    # Identify voltage columns and metadata columns
    voltage_columns = []
    metadata_columns = []

    # First, filter out columns that should be skipped
    columns_to_skip = ['path']

    # Skip unnamed columns and columns with strange formats
    for col in df.columns:
        if 'unnamed' in str(col).lower() or 'unnamed:' in str(col).lower():
            columns_to_skip.append(col)
            print(f"Skipping unnamed column: {col}")
        # Skip columns with strange scientific notation formats that aren't valid voltages
        elif re.match(r'^-?\d+e[-+]\d+\.\d+', str(col)):
            columns_to_skip.append(col)
            print(f"Skipping column with strange format: {col}")

    # First identify important metadata columns
    important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
    for col in df.columns:
        if col in columns_to_skip:
            continue

        if str(col).lower() in [meta.lower() for meta in important_metadata]:
            # Found an important metadata column - use the original column name
            metadata_columns.append(col)
            print(f"Found important metadata column: {col}")

    # Then process remaining columns
    for col in df.columns:
        # Skip columns we've already processed
        if col in metadata_columns or col in columns_to_skip:
            continue

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
                        # Only add as metadata if it's not a column to skip
                        if col not in columns_to_skip:
                            metadata_columns.append(col)
            # If not a voltage column, it might be a metadata column
            elif col not in columns_to_skip:
                metadata_columns.append(col)
                print(f"Found additional metadata column: {col}")

    # Create a list to store features for each sample
    all_features = []

    # Process each sample (row) individually
    for row_idx, row in df.iterrows():
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

            # Add row index as a feature for tracking
            sample_features['row_index'] = row_idx

            # Add metadata
            for col in metadata_columns:
                if col in row.index:
                    sample_features[col] = row[col]

            # Add to the list of all features
            all_features.append(sample_features)

    # Convert to DataFrame
    if all_features:
        features_df = pd.DataFrame(all_features)

        # Filter out problematic features if requested
        if filter_features:
            # Identify target column for supervised importance calculation if available
            target_column = None
            for col in ['concentration', 'antibiotic']:
                if col in features_df.columns:
                    target_column = col
                    print(f"Using {col} as target column for feature importance calculation")
                    break

            # Apply dynamic feature importance filtering
            features_df = filter_low_importance_features(features_df, threshold=0.4, target_column=target_column)

            print(f"Filtered features from {len(all_features[0]) if all_features else 0} to {len(features_df.columns)} columns")

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

    # Derivative features - removed as they are not needed for electrochemical analysis

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
    "derivative": "Features based on the rate of change of current - removed as not needed",
    "area": "Features related to the area under the voltammogram"
}


def filter_low_importance_features(feature_df, threshold=0.4, target_column=None):
    """
    Filter out features with importance below the threshold based on calculated feature importance

    Args:
        feature_df: DataFrame with features
        threshold: Importance threshold (features with importance below this will be removed)
        target_column: Target column for supervised importance calculation (if None, use unsupervised methods)

    Returns:
        DataFrame with low importance features removed
    """
    # Create a copy of the dataframe
    filtered_df = feature_df.copy()

    # Identify metadata columns to exclude from feature importance calculation
    metadata_columns = ['row_index', 'concentration', 'antibiotic', 'path']
    feature_columns = [col for col in filtered_df.columns if col not in metadata_columns]

    # Always exclude derivative features as they're not needed for electrochemical analysis
    derivative_features = [col for col in feature_columns if col.startswith('derivative_')]
    for feature in derivative_features:
        if feature in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=[feature])
            print(f"Removed derivative feature: {feature}")

    # Update feature columns list after removing derivative features
    feature_columns = [col for col in feature_columns if col not in derivative_features]

    if len(feature_columns) == 0:
        print("No features left after removing derivative features")
        return filtered_df

    # Calculate feature importance
    if target_column is not None and target_column in filtered_df.columns:
        # Supervised importance calculation using Random Forest
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Prepare data
            X = filtered_df[feature_columns]
            y = filtered_df[target_column]

            # Handle missing values
            X = X.fillna(0)

            # Train a simple model to get feature importance
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            # Get feature importance
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_columns, importances))

            print("\nFeature importance (supervised):\n")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")

            # Remove features with importance below threshold
            low_importance_features = [feature for feature, importance in feature_importance.items()
                                     if importance < threshold]

            for feature in low_importance_features:
                if feature in filtered_df.columns:
                    filtered_df = filtered_df.drop(columns=[feature])
                    print(f"Removed low importance feature: {feature} (importance: {feature_importance[feature]:.4f})")

        except Exception as e:
            print(f"Error calculating supervised feature importance: {e}")
            print("Falling back to unsupervised methods")
            # Fall back to unsupervised methods if supervised fails
            target_column = None

    if target_column is None:
        # Unsupervised importance calculation using variance
        try:
            # Calculate variance for each feature
            variances = filtered_df[feature_columns].var()

            # Normalize variances to [0, 1] scale
            if variances.max() > 0:
                normalized_variances = variances / variances.max()
            else:
                normalized_variances = variances

            feature_importance = dict(zip(feature_columns, normalized_variances))

            print("\nFeature importance (unsupervised - variance):\n")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")

            # Remove features with importance below threshold
            low_importance_features = [feature for feature, importance in feature_importance.items()
                                     if importance < threshold]

            for feature in low_importance_features:
                if feature in filtered_df.columns:
                    filtered_df = filtered_df.drop(columns=[feature])
                    print(f"Removed low importance feature: {feature} (importance: {feature_importance[feature]:.4f})")

        except Exception as e:
            print(f"Error calculating unsupervised feature importance: {e}")

    return filtered_df

def extract_raw_features_from_samples(df):
    """
    Extract raw features from voltammetric data on a sample-by-sample basis.
    Instead of calculating derived features, this function uses the raw current values
    at each potential as features directly.

    This function processes a dataset where:
    - Each row represents a sample
    - Columns include voltage points (e.g., -0.8, -0.79, etc.)
    - Additional metadata columns like 'concentration' and 'antibiotic' are present

    Args:
        df: DataFrame with voltage columns and metadata columns

    Returns:
        DataFrame with raw features (current values) for each sample and preserved metadata
    """
    print("\n==== EXTRACTING RAW VOLTAGE FEATURES ====")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns (first 5): {list(df.columns)[:5]}")

    # Identify voltage columns and metadata columns
    voltage_columns = []
    metadata_columns = []

    # First, filter out columns that should be skipped
    columns_to_skip = ['path']

    # Skip unnamed columns and columns with strange formats
    for col in df.columns:
        if 'unnamed' in str(col).lower() or 'unnamed:' in str(col).lower():
            columns_to_skip.append(col)
            print(f"Skipping unnamed column: {col}")
        # Skip columns with strange scientific notation formats that aren't valid voltages
        elif re.match(r'^-?\d+e[-+]\d+\.\d+', str(col)):
            columns_to_skip.append(col)
            print(f"Skipping column with strange format: {col}")

    # First identify important metadata columns
    important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
    for col in df.columns:
        if col in columns_to_skip:
            continue

        if str(col).lower() in [meta.lower() for meta in important_metadata]:
            # Found an important metadata column - use the original column name
            metadata_columns.append(col)
            print(f"Found important metadata column: {col}")

    # Then process remaining columns
    for col in df.columns:
        # Skip columns we've already processed
        if col in metadata_columns or col in columns_to_skip:
            continue

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
                        # Only add as metadata if it's not a column to skip
                        if col not in columns_to_skip:
                            metadata_columns.append(col)
            # If not a voltage column, it might be a metadata column
            elif col not in columns_to_skip:
                metadata_columns.append(col)
                print(f"Found additional metadata column: {col}")

    print(f"Found {len(voltage_columns)} voltage columns and {len(metadata_columns)} metadata columns")

    # Create a new DataFrame with raw features
    # Start with metadata columns
    if metadata_columns:
        raw_features_df = df[metadata_columns].copy()
    else:
        raw_features_df = pd.DataFrame(index=df.index)

    # Add row index as a feature for tracking
    raw_features_df['row_index'] = df.index

    if not voltage_columns:
        print("WARNING: No voltage columns found in the data!")
        return raw_features_df

    # Sort voltage columns by their numerical value for consistency
    try:
        sorted_voltage_columns = sorted(voltage_columns, key=lambda x:
                                       float(x.split('.')[0] + '.' + x.split('.')[1])
                                       if '.' in str(x) and len(x.split('.')) > 1
                                       else float(x))
    except Exception as e:
        print(f"Error sorting voltage columns: {e}")
        print("Using unsorted voltage columns instead")
        sorted_voltage_columns = voltage_columns

    # Add voltage columns with renamed headers for clarity
    voltage_count = 0
    for i, col in enumerate(sorted_voltage_columns):
        # Rename the column to a more descriptive name
        try:
            if isinstance(col, str) and '.' in col:
                parts = col.split('.')
                voltage = float(f"{parts[0]}.{parts[1]}")
            else:
                voltage = float(col)

            # Format the voltage to 3 decimal places
            new_col_name = f"V_{voltage:.3f}"
            raw_features_df[new_col_name] = df[col]
            voltage_count += 1
        except (ValueError, TypeError) as e:
            # Skip if conversion fails
            print(f"Error converting column {col} to voltage: {e}")
            continue

    print(f"Created raw features DataFrame with {len(raw_features_df.columns)} columns ({voltage_count} voltage features)")
    print(f"Output DataFrame shape: {raw_features_df.shape}")
    print("===================================\n")
    return raw_features_df


def apply_pca_to_wide_data(df, n_components=None, variance_threshold=0.95):
    """
    Apply PCA to wide-format voltammetric data to reduce dimensionality.

    Args:
        df: DataFrame with voltage columns and metadata columns
        n_components: Number of principal components to keep (if None, use variance_threshold)
        variance_threshold: Minimum cumulative explained variance ratio to retain (default: 0.95)

    Returns:
        DataFrame with PCA components and metadata columns, PCA model
    """
    # Identify voltage columns and metadata columns
    voltage_columns = []
    metadata_columns = []

    # First, filter out columns that should be skipped
    columns_to_skip = ['path']

    # Skip unnamed columns and columns with strange formats
    for col in df.columns:
        if 'unnamed' in str(col).lower() or 'unnamed:' in str(col).lower():
            columns_to_skip.append(col)
            print(f"Skipping unnamed column: {col}")
        # Skip columns with strange scientific notation formats that aren't valid voltages
        elif re.match(r'^-?\d+e[-+]\d+\.\d+', str(col)):
            columns_to_skip.append(col)
            print(f"Skipping column with strange format: {col}")

    # First identify important metadata columns
    important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
    for col in df.columns:
        if col in columns_to_skip:
            continue

        if str(col).lower() in [meta.lower() for meta in important_metadata]:
            # Found an important metadata column - use the original column name
            metadata_columns.append(col)
            print(f"Found important metadata column: {col}")

    # Then process remaining columns
    for col in df.columns:
        # Skip columns we've already processed
        if col in metadata_columns or col in columns_to_skip:
            continue

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
                        # Only add as metadata if it's not a column to skip
                        if col not in columns_to_skip:
                            metadata_columns.append(col)
            # If not a voltage column, it might be a metadata column
            elif col not in columns_to_skip:
                metadata_columns.append(col)
                print(f"Found additional metadata column: {col}")

    print(f"Found {len(voltage_columns)} voltage columns and {len(metadata_columns)} metadata columns")

    # Sort voltage columns by their numerical value for consistency
    try:
        sorted_voltage_columns = sorted(voltage_columns, key=lambda x:
                                       float(x.split('.')[0] + '.' + x.split('.')[1])
                                       if '.' in str(x) and len(x.split('.')) > 1
                                       else float(x))
        voltage_columns = sorted_voltage_columns
    except Exception as e:
        print(f"Error sorting voltage columns: {e}")

    # Extract voltage data for PCA
    X = df[voltage_columns].fillna(0).values

    # Initialize PCA
    if n_components is None:
        # Start with all components and then filter based on explained variance
        pca = PCA()
    else:
        # Use specified number of components
        pca = PCA(n_components=n_components)

    # Fit and transform the data
    X_pca = pca.fit_transform(X)

    # If n_components is None, determine number of components based on variance threshold
    if n_components is None:
        # Calculate cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find number of components that explain at least variance_threshold of variance
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Ensure we have at least 3 components (even if first component explains >95% variance)
        n_components = max(3, n_components)

        # Print information about variance explained
        print(f"Selected {n_components} components to explain at least {variance_threshold*100:.1f}% of variance")
        print(f"First component explains {explained_variance_ratio[0]*100:.2f}% of variance")
        if len(explained_variance_ratio) > 1:
            print(f"Second component explains {explained_variance_ratio[1]*100:.2f}% of variance")
        if len(explained_variance_ratio) > 2:
            print(f"Third component explains {explained_variance_ratio[2]*100:.2f}% of variance")

        # Create a new PCA model with the determined number of components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

    # Create a new DataFrame with PCA components
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

    # Add metadata columns
    for col in metadata_columns:
        pca_df[col] = df[col].values

    # Add row index as a feature for tracking
    pca_df['row_index'] = df.index

    # Print explained variance information
    print(f"PCA with {n_components} components explains {pca.explained_variance_ratio_.sum()*100:.2f}% of variance")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio*100:.2f}% of variance")

    return pca_df, pca


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