import numpy as np
import pandas as pd
from scipy import integrate, fft
import re
from sklearn.decomposition import PCA


# Dictionary of feature categories with descriptions
FEATURE_CATEGORIES = {
    "pca": "Principal Component Analysis features for dimensionality reduction",
    "fft": "Fast Fourier Transform features in frequency domain",
    "windowed_integral": "Windowed slicing integral for area-based features extraction"
}


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


def apply_fft_to_wide_data(df, n_components=10):
    """
    Áp dụng Fast Fourier Transform cho dữ liệu voltammetry dạng rộng để phân tích tần số 
    và giảm chiều dữ liệu.

    Args:
        df: DataFrame với các cột điện thế và cột metadata
        n_components: Số thành phần FFT cần giữ lại (mặc định: 10)

    Returns:
        DataFrame với các thành phần FFT và cột metadata, thông tin biến đổi
    """
    # Xác định các cột điện thế và cột metadata
    voltage_columns = []
    metadata_columns = []
    columns_to_skip = ['path']

    # Xác định các cột metadata quan trọng
    important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
    for col in df.columns:
        if col in columns_to_skip:
            continue

        if str(col).lower() in [meta.lower() for meta in important_metadata]:
            metadata_columns.append(col)
            print(f"Đã tìm thấy cột metadata quan trọng: {col}")

    # Xử lý các cột còn lại
    for col in df.columns:
        if col in metadata_columns or col in columns_to_skip:
            continue

        try:
            float(col)
            voltage_columns.append(col)
        except ValueError:
            if isinstance(col, str) and re.match(r'^-?\d+\.\d+(\.\d+)*$', col):
                parts = col.split('.')
                if len(parts) >= 2:
                    try:
                        float(f"{parts[0]}.{parts[1]}")
                        voltage_columns.append(col)
                    except ValueError:
                        if col not in columns_to_skip:
                            metadata_columns.append(col)
            elif col not in columns_to_skip:
                metadata_columns.append(col)

    print(f"Đã tìm thấy {len(voltage_columns)} cột điện thế và {len(metadata_columns)} cột metadata")

    # Sắp xếp cột điện thế theo giá trị số để đảm bảo tính nhất quán
    try:
        sorted_voltage_columns = sorted(voltage_columns, key=lambda x:
                                      float(x.split('.')[0] + '.' + x.split('.')[1])
                                      if '.' in str(x) and len(x.split('.')) > 1
                                      else float(x))
        voltage_columns = sorted_voltage_columns
    except Exception as e:
        print(f"Lỗi khi sắp xếp cột điện thế: {e}")

    # Trích xuất dữ liệu điện thế cho FFT
    X = df[voltage_columns].fillna(0).values

    # Số mẫu và số điểm dữ liệu
    n_samples, n_points = X.shape
    
    # Danh sách lưu tất cả các thành phần FFT
    fft_features = []
    
    # Áp dụng FFT trên mỗi mẫu
    for i in range(n_samples):
        # Lấy tín hiệu điện hóa cho mẫu hiện tại
        signal = X[i]
        
        # Tính toán FFT
        fft_result = fft.fft(signal)
        fft_magnitude = np.abs(fft_result)
        
        # Lấy n_components thành phần đầu tiên
        # (bỏ qua thành phần DC ở vị trí 0)
        fft_components = fft_magnitude[1:n_components+1]
        
        # Thêm vào danh sách kết quả
        fft_features.append(fft_components)
    
    # Chuyển đổi thành mảng NumPy
    fft_features = np.array(fft_features)
    
    # Tạo DataFrame mới với các thành phần FFT
    fft_df = pd.DataFrame(fft_features, columns=[f"FFT{i+1}" for i in range(n_components)])
    
    # Thêm cột metadata
    for col in metadata_columns:
        fft_df[col] = df[col].values
    
    # Thêm chỉ số hàng để theo dõi
    fft_df['row_index'] = df.index
    
    # In thông tin về biến đổi
    print(f"Biến đổi FFT với {n_components} thành phần trích xuất từ {n_points} điểm dữ liệu")
    
    # Tạo từ điển thông tin về FFT để lưu trữ
    fft_info = {
        "n_components": n_components,
        "n_points": n_points,
        "voltage_columns": voltage_columns
    }
    
    return fft_df, fft_info


def apply_windowed_integral_to_wide_data(df, n_windows=10, normalize=True):
    """
    Áp dụng phương pháp chia cửa sổ tích phân cho dữ liệu voltammetry dạng rộng.
    
    Phương pháp này chia tín hiệu thành các cửa sổ đều nhau và tính diện tích dưới đường cong
    trong từng cửa sổ, tạo ra vector đặc trưng cho mỗi mẫu.
    
    Args:
        df: DataFrame với các cột điện thế và cột metadata
        n_windows: Số lượng cửa sổ cần chia (mặc định: 10)
        normalize: Chuẩn hóa diện tích theo độ rộng cửa sổ (mặc định: True)
        
    Returns:
        DataFrame với các đặc trưng diện tích trong mỗi cửa sổ và cột metadata, thông tin về phương pháp
    """
    # Xác định các cột điện thế và cột metadata
    voltage_columns = []
    metadata_columns = []
    columns_to_skip = ['path']

    # Xác định các cột metadata quan trọng
    important_metadata = ['concentration', 'antibiotic', 'label', 'class', 'target']
    for col in df.columns:
        if col in columns_to_skip:
            continue

        if str(col).lower() in [meta.lower() for meta in important_metadata]:
            metadata_columns.append(col)
            print(f"Đã tìm thấy cột metadata quan trọng: {col}")

    # Xử lý các cột còn lại
    for col in df.columns:
        if col in metadata_columns or col in columns_to_skip:
            continue

        try:
            float(col)
            voltage_columns.append(col)
        except ValueError:
            if isinstance(col, str) and re.match(r'^-?\d+\.\d+(\.\d+)*$', col):
                parts = col.split('.')
                if len(parts) >= 2:
                    try:
                        float(f"{parts[0]}.{parts[1]}")
                        voltage_columns.append(col)
                    except ValueError:
                        if col not in columns_to_skip:
                            metadata_columns.append(col)
            elif col not in columns_to_skip:
                metadata_columns.append(col)

    print(f"Đã tìm thấy {len(voltage_columns)} cột điện thế và {len(metadata_columns)} cột metadata")

    # Sắp xếp cột điện thế theo giá trị số để đảm bảo tính nhất quán
    try:
        sorted_voltage_columns = sorted(voltage_columns, key=lambda x:
                                      float(x.split('.')[0] + '.' + x.split('.')[1])
                                      if '.' in str(x) and len(x.split('.')) > 1
                                      else float(x))
        voltage_columns = sorted_voltage_columns
    except Exception as e:
        print(f"Lỗi khi sắp xếp cột điện thế: {e}")
    
    # Chuyển đổi các nhãn điện thế thành giá trị số
    voltage_values = []
    for col in voltage_columns:
        try:
            if isinstance(col, str) and '.' in col:
                parts = col.split('.')
                voltage = float(f"{parts[0]}.{parts[1]}")
            else:
                voltage = float(col)
            voltage_values.append(voltage)
        except (ValueError, TypeError) as e:
            print(f"Lỗi khi chuyển đổi cột {col} thành giá trị điện thế: {e}")
    
    # Trích xuất dữ liệu điện thế
    X = df[voltage_columns].fillna(0).values
    
    # Số mẫu
    n_samples, n_points = X.shape
    
    # Danh sách lưu các đặc trưng diện tích cửa sổ
    window_features = []
    
    # Tính toán số điểm trong mỗi cửa sổ
    points_per_window = n_points // n_windows
    extra_points = n_points % n_windows
    
    # Diện tích tích phân cho mỗi mẫu
    for i in range(n_samples):
        # Lấy tín hiệu cho mẫu hiện tại
        signal = X[i]
        x_values = np.array(voltage_values)
        
        # Tính toán diện tích cho mỗi cửa sổ
        window_areas = []
        start_idx = 0
        
        for j in range(n_windows):
            # Điều chỉnh kích thước cửa sổ cuối cùng nếu cần
            if j < extra_points:
                end_idx = start_idx + points_per_window + 1
            else:
                end_idx = start_idx + points_per_window
                
            if end_idx > n_points:
                end_idx = n_points
            
            # Trích xuất cửa sổ tín hiệu
            window_signal = signal[start_idx:end_idx]
            window_x = x_values[start_idx:end_idx]
            
            if len(window_signal) > 1:
                # Tính diện tích sử dụng quy tắc hình thang
                area = integrate.trapz(window_signal, window_x)
                
                # Chuẩn hóa theo chiều rộng cửa sổ nếu cần
                if normalize and (window_x[-1] - window_x[0]) != 0:
                    area = area / (window_x[-1] - window_x[0])
                
                window_areas.append(area)
            else:
                # Trường hợp chỉ có 1 điểm hoặc không có điểm nào
                window_areas.append(0)
            
            # Cập nhật chỉ số bắt đầu cho cửa sổ tiếp theo
            start_idx = end_idx
        
        window_features.append(window_areas)
    
    # Chuyển đổi thành mảng NumPy
    window_features = np.array(window_features)
    
    # Tạo DataFrame mới với các đặc trưng diện tích cửa sổ
    window_df = pd.DataFrame(window_features, columns=[f"Window{i+1}" for i in range(n_windows)])
    
    # Thêm cột metadata
    for col in metadata_columns:
        window_df[col] = df[col].values
    
    # Thêm chỉ số hàng để theo dõi
    window_df['row_index'] = df.index
    
    # In thông tin về biến đổi
    print(f"Đã trích xuất {n_windows} đặc trưng diện tích từ {n_points} điểm dữ liệu bằng phương pháp chia cửa sổ tích phân")
    
    # Tạo từ điển thông tin để lưu trữ
    window_info = {
        "n_windows": n_windows,
        "n_points": n_points,
        "normalized": normalize,
        "voltage_columns": voltage_columns
    }
    
    return window_df, window_info


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
        feature_importance = dict(zip(feature_df.columns, importances))

        # Select features above threshold
        selected_features = [feature for feature, importance in feature_importance.items()
                            if importance > threshold]

    else:
        # Default: return all features
        selected_features = feature_df.columns.tolist()

    return selected_features