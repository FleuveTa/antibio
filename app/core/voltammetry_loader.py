import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.signal import find_peaks
from scipy.signal import detrend


def load_voltammetry_data(file_path, data_type="CV"):
    """
    Load voltammetric data from a file.
    Supports CV (Cyclic Voltammetry) and DPV (Differential Pulse Voltammetry).
    
    Args:
        file_path: Path to the data file
        data_type: Type of voltammetry data (CV or DPV)
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        # Try loading with comma separator first
        try:
            df = pd.read_csv(file_path)
        except:
            # Try with semicolon separator
            df = pd.read_csv(file_path, sep=';')
        
        # Check if we have the required columns based on data type
        if data_type == "CV":
            required_cols = ["Potential", "Current"]
        elif data_type == "DPV":
            required_cols = ["Potential", "Current"]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Validate columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try to find columns with units in parentheses
            for col in missing_cols:
                potential_match = next((c for c in df.columns if col in c), None)
                if potential_match:
                    df = df.rename(columns={potential_match: col})
            
            # Final check
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Could not find required columns: {', '.join(missing_cols)}")
        
        # Convert string values to numeric if needed
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata
        df.attrs["data_type"] = data_type
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading voltammetry data: {str(e)}")


def preprocess_voltammetry_data(df, data_type, options):
    """
    Preprocess voltammetric data with the specified options.
    
    Args:
        df: DataFrame with voltammetry data
        data_type: Type of voltammetry data (CV or DPV)
        options: Dictionary of preprocessing options
        
    Returns:
        Processed DataFrame and a dictionary of transformers
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
        if window_length >= 3:  # Minimum window length for Savitzky-Golay filter
            processed_df["Current"] = savgol_filter(processed_df["Current"], window_length, 3)
    
    # Baseline correction
    if options.get("baseline_correction", False):
        processed_df["Current"] = detrend(processed_df["Current"])
    
    # Peak detection
    if options.get("peak_detection", False):
        # For CV data, we need to handle forward and reverse scans separately
        if data_type == "CV":
            # Find the turning point (usually the maximum or minimum potential)
            turning_idx = processed_df["Potential"].idxmax() if processed_df["Potential"].iloc[0] < processed_df["Potential"].iloc[-1] else processed_df["Potential"].idxmin()
            
            # Split into forward and reverse scans
            forward_scan = processed_df.iloc[:turning_idx+1]
            reverse_scan = processed_df.iloc[turning_idx:]
            
            # Find peaks in each scan
            forward_peaks, _ = find_peaks(forward_scan["Current"], height=0)
            reverse_peaks, _ = find_peaks(reverse_scan["Current"], height=0)
            
            # Mark peaks in the original dataframe
            processed_df["Peaks"] = False
            processed_df.iloc[forward_peaks, processed_df.columns.get_loc("Peaks")] = True
            processed_df.iloc[turning_idx + reverse_peaks, processed_df.columns.get_loc("Peaks")] = True
            
        else:  # DPV or other techniques
            peaks, _ = find_peaks(processed_df["Current"], height=0)
            processed_df["Peaks"] = False
            processed_df.iloc[peaks, processed_df.columns.get_loc("Peaks")] = True
    
    return processed_df, transformers


def extract_voltammetry_features(df, data_type):
    """
    Extract features specific to voltammetry data
    
    Args:
        df: DataFrame with voltammetry data
        data_type: Type of voltammetry data (CV or DPV)
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Basic statistical features
    features["mean_current"] = df["Current"].mean()
    features["std_current"] = df["Current"].std()
    features["min_current"] = df["Current"].min()
    features["max_current"] = df["Current"].max()
    features["current_range"] = features["max_current"] - features["min_current"]
    
    # Peak features
    if "Peaks" in df.columns:
        peak_indices = df[df["Peaks"]].index
        if len(peak_indices) > 0:
            peak_currents = df.loc[peak_indices, "Current"]
            peak_potentials = df.loc[peak_indices, "Potential"]
            
            features["num_peaks"] = len(peak_indices)
            features["max_peak_current"] = peak_currents.max()
            features["max_peak_potential"] = peak_potentials[peak_currents.argmax()]
            
            # Peak-to-peak distances if multiple peaks
            if len(peak_indices) >= 2:
                peak_distances = np.diff(peak_potentials)
                features["mean_peak_distance"] = peak_distances.mean()
                features["std_peak_distance"] = peak_distances.std()
    
    # Area under curve
    features["auc"] = np.trapz(df["Current"], df["Potential"])
    
    # CV-specific features
    if data_type == "CV":
        # Find the turning point
        turning_idx = df["Potential"].idxmax() if df["Potential"].iloc[0] < df["Potential"].iloc[-1] else df["Potential"].idxmin()
        
        # Split into forward and reverse scans
        forward_scan = df.iloc[:turning_idx+1]
        reverse_scan = df.iloc[turning_idx:]
        
        # Calculate features for each scan
        features["forward_max_current"] = forward_scan["Current"].max()
        features["reverse_max_current"] = reverse_scan["Current"].max()
        features["forward_min_current"] = forward_scan["Current"].min()
        features["reverse_min_current"] = reverse_scan["Current"].min()
        
        # Calculate peak separation if peaks are detected
        if "Peaks" in df.columns:
            forward_peaks = forward_scan[forward_scan["Peaks"]]
            reverse_peaks = reverse_scan[reverse_scan["Peaks"]]
            
            if len(forward_peaks) > 0 and len(reverse_peaks) > 0:
                # Find the main peaks (maximum current)
                main_forward_peak = forward_peaks.loc[forward_peaks["Current"].idxmax()]
                main_reverse_peak = reverse_peaks.loc[reverse_peaks["Current"].idxmax()]
                
                # Calculate peak separation
                features["peak_separation"] = abs(main_forward_peak["Potential"] - main_reverse_peak["Potential"])
                features["peak_current_ratio"] = abs(main_forward_peak["Current"] / main_reverse_peak["Current"])
    
    # DPV-specific features
    elif data_type == "DPV":
        # Find the main peak (maximum current)
        if "Peaks" in df.columns and df["Peaks"].any():
            peaks = df[df["Peaks"]]
            main_peak = peaks.loc[peaks["Current"].idxmax()]
            
            features["main_peak_potential"] = main_peak["Potential"]
            features["main_peak_current"] = main_peak["Current"]
            
            # Calculate peak width (at half height)
            half_height = main_peak["Current"] / 2
            
            # Find points closest to half height on both sides of the peak
            peak_idx = main_peak.name
            left_side = df.iloc[:peak_idx]
            right_side = df.iloc[peak_idx:]
            
            if not left_side.empty and not right_side.empty:
                left_side["diff"] = abs(left_side["Current"] - half_height)
                right_side["diff"] = abs(right_side["Current"] - half_height)
                
                left_point = left_side.loc[left_side["diff"].idxmin()]
                right_point = right_side.loc[right_side["diff"].idxmin()]
                
                features["peak_width"] = abs(right_point["Potential"] - left_point["Potential"])
    
    return features
