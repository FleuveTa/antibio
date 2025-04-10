import numpy as np
import pandas as pd
from scipy import signal, integrate


def extract_voltammetric_features(df):
    """
    Extract features from voltammetric data
    """
    features = {}
    
    # Basic statistical features
    features["mean_current"] = df["Current"].mean()
    features["std_current"] = df["Current"].std()
    features["min_current"] = df["Current"].min()
    features["max_current"] = df["Current"].max()
    
    # Peak features
    if "Peaks" in df.columns:
        peak_indices = df[df["Peaks"]].index
        if len(peak_indices) > 0:
            peak_currents = df.loc[peak_indices, "Current"]
            peak_potentials = df.loc[peak_indices, "Potential"]
            
            features["num_peaks"] = len(peak_indices)
            features["max_peak_current"] = peak_currents.max()
            features["max_peak_potential"] = peak_potentials[peak_currents.argmax()]
            
            # Peak-to-peak distances
            if len(peak_indices) >= 2:
                peak_distances = np.diff(peak_potentials)
                features["mean_peak_distance"] = peak_distances.mean()
                features["std_peak_distance"] = peak_distances.std()
    
    # Area under curve
    if "Potential" in df.columns and "Current" in df.columns:
        area = integrate.trapz(df["Current"], df["Potential"])
        features["auc"] = area
    
    # More complex features could be added
    
    return features


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


def select_features(feature_df, target, method="correlation", threshold=0.2):
    """
    Select relevant features based on different methods
    """
    if method == "correlation":
        # Calculate correlation with target
        correlations = feature_df.corrwith(target).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        
    elif method == "variance":
        # Remove low-variance features
        variances = feature_df.var()
        selected_features = variances[variances > threshold].index.tolist()
        
    # Could add more methods: RFE, PCA, etc.
    
    return selected_features