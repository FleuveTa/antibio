import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks, detrend
from scipy.stats import zscore
from sklearn.impute import SimpleImputer


def load_voltammetry_data(file_path, data_type="CV"):
    """
    Tải dữ liệu voltammetry từ file.
    Hỗ trợ CV (Cyclic Voltammetry) và DPV (Differential Pulse Voltammetry).
    
    Args:
        file_path: Đường dẫn đến file dữ liệu
        data_type: Loại dữ liệu voltammetry (CV hoặc DPV)
        
    Returns:
        DataFrame chứa dữ liệu đã tải
    """
    try:
        # Đọc file với dấu phẩy làm phân cách
        df = pd.read_csv(file_path, sep=',', error_bad_lines=False)
        
        # Kiểm tra các cột cần thiết
        required_cols = ["Potential", "Current"]
        
        # Kiểm tra và đổi tên cột nếu cần
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                potential_match = next((c for c in df.columns if col in c.lower()), None)
                if potential_match:
                    df = df.rename(columns={potential_match: col})
            
            # Kiểm tra cuối cùng
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Không tìm thấy các cột cần thiết: {', '.join(missing_cols)}")
        
        # Chuyển đổi giá trị string thành số
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Thêm thông tin metadata
        df.attrs["data_type"] = data_type
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("File không chứa dữ liệu")
    except pd.errors.ParserError:
        raise ValueError("Lỗi khi phân tích file")
    except Exception as e:
        raise ValueError(f"Lỗi khi tải dữ liệu voltammetry: {str(e)}")


def preprocess_voltammetry_data(df, data_type, options):
    """
    Xử lý dữ liệu voltammetry với các tùy chọn đã chỉ định.
    
    Args:
        df: DataFrame chứa dữ liệu voltammetry
        data_type: Loại dữ liệu voltammetry (CV hoặc DPV)
        options: Từ điển chứa các tùy chọn xử lý
        
    Returns:
        DataFrame đã xử lý và từ điển các transformer
    """
    processed_df = df.copy()
    transformers = {}
    
    # Xử lý giá trị thiếu
    if options.get("fill_missing", False):
        imputer = SimpleImputer(strategy='mean')
        processed_df[["Potential", "Current"]] = imputer.fit_transform(processed_df[["Potential", "Current"]])
        transformers["impute"] = imputer
    
    # Xóa giá trị ngoại lai
    if options.get("remove_outliers", False):
        z_scores = np.abs(zscore(processed_df[["Potential", "Current"]]))
        mask = (z_scores < 3).all(axis=1)
        outlier_count = (~mask).sum()
        if outlier_count > 0:
            processed_df = processed_df[mask]
    
    # Làm mịn tín hiệu
    if options.get("smooth", False):
        window_length = min(51, len(processed_df) // 2 * 2 - 1)
        if window_length >= 3:
            processed_df["Current"] = savgol_filter(processed_df["Current"], window_length, 3)
    
    # Hiệu chỉnh đường cơ bản
    if options.get("baseline_correction", False):
        processed_df["Current"] = detrend(processed_df["Current"])
    
    # Phát hiện đỉnh
    if options.get("peak_detection", False):
        if data_type == "CV":
            turning_idx = processed_df["Potential"].idxmax() if processed_df["Potential"].iloc[0] < processed_df["Potential"].iloc[-1] else processed_df["Potential"].idxmin()
            
            forward_scan = processed_df.iloc[:turning_idx+1]
            reverse_scan = processed_df.iloc[turning_idx:]
            
            forward_peaks, _ = find_peaks(forward_scan["Current"], height=0)
            reverse_peaks, _ = find_peaks(reverse_scan["Current"], height=0)
            
            processed_df["Peaks"] = False
            processed_df.iloc[forward_peaks, processed_df.columns.get_loc("Peaks")] = True
            processed_df.iloc[turning_idx + reverse_peaks, processed_df.columns.get_loc("Peaks")] = True
            
        else:
            peaks, _ = find_peaks(processed_df["Current"], height=0)
            processed_df["Peaks"] = False
            processed_df.iloc[peaks, processed_df.columns.get_loc("Peaks")] = True
    
    return processed_df, transformers


def extract_voltammetry_features(df, data_type):
    """
    Trích xuất các đặc trưng từ dữ liệu voltammetry
    
    Args:
        df: DataFrame chứa dữ liệu voltammetry
        data_type: Loại dữ liệu voltammetry (CV hoặc DPV)
        
    Returns:
        Từ điển chứa các đặc trưng đã trích xuất
    """
    features = {}
    
    # Các đặc trưng thống kê cơ bản
    features["mean_current"] = df["Current"].mean()
    features["std_current"] = df["Current"].std()
    features["min_current"] = df["Current"].min()
    features["max_current"] = df["Current"].max()
    features["current_range"] = features["max_current"] - features["min_current"]
    
    # Các đặc trưng về đỉnh
    if "Peaks" in df.columns:
        peak_indices = df[df["Peaks"]].index
        if len(peak_indices) > 0:
            peak_currents = df.loc[peak_indices, "Current"]
            peak_potentials = df.loc[peak_indices, "Potential"]
            
            features["num_peaks"] = len(peak_indices)
            features["max_peak_current"] = peak_currents.max()
            features["max_peak_potential"] = peak_potentials[peak_currents.argmax()]
            
            if len(peak_indices) >= 2:
                peak_distances = np.diff(peak_potentials)
                features["mean_peak_distance"] = peak_distances.mean()
                features["std_peak_distance"] = peak_distances.std()
    
    # Diện tích dưới đường cong
    features["auc"] = np.trapz(df["Current"], df["Potential"])
    
    # Các đặc trưng riêng của CV
    if data_type == "CV":
        turning_idx = df["Potential"].idxmax() if df["Potential"].iloc[0] < df["Potential"].iloc[-1] else df["Potential"].idxmin()
        
        forward_scan = df.iloc[:turning_idx+1]
        reverse_scan = df.iloc[turning_idx:]
        
        features["forward_max_current"] = forward_scan["Current"].max()
        features["reverse_max_current"] = reverse_scan["Current"].max()
        features["forward_min_current"] = forward_scan["Current"].min()
        features["reverse_min_current"] = reverse_scan["Current"].min()
        
        if "Peaks" in df.columns:
            forward_peaks = forward_scan[forward_scan["Peaks"]]
            reverse_peaks = reverse_scan[reverse_scan["Peaks"]]
            
            if len(forward_peaks) > 0 and len(reverse_peaks) > 0:
                main_forward_peak = forward_peaks.loc[forward_peaks["Current"].idxmax()]
                main_reverse_peak = reverse_peaks.loc[reverse_peaks["Current"].idxmax()]
                
                # Tính khoảng cách giữa các đỉnh
                peak_separation = abs(main_forward_peak["Potential"] - main_reverse_peak["Potential"])
                features["peak_separation"] = peak_separation
                
                # Tính tỷ lệ giữa các đỉnh
                peak_ratio = main_forward_peak["Current"] / main_reverse_peak["Current"]
                features["peak_ratio"] = peak_ratio
    
    return features
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
