import numpy as np
import pandas as pd
import re
from scipy.signal import savgol_filter, find_peaks, detrend
from scipy.stats import zscore
from sklearn.impute import SimpleImputer


def preprocess_data(df, options=None):
    """
    Xử lý dữ liệu voltammetry với các tùy chọn đã chỉ định.
    Tập trung vào làm sạch dữ liệu, xử lý giá trị thiếu và giảm nhiễu.

    Args:
        df: DataFrame chứa dữ liệu voltammetry
        options: Từ điển chứa các tùy chọn xử lý

    Returns:
        DataFrame đã xử lý và từ điển các transformer
    """
    processed_df = df.copy()
    transformers = {}

    # Bảo toàn các cột metadata
    metadata_columns = [col for col in processed_df.columns
                       if col not in ["Potential", "Current"]]

    # Xử lý giá trị thiếu
    if options.get("fill_missing", False):
        numeric_cols = ["Potential", "Current"]
        numeric_data = processed_df[numeric_cols].copy()
        
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numeric_data)
        processed_df[numeric_cols] = imputed_data
        transformers["impute"] = imputer
        
        print(f"Đã điền {imputer.statistics_.sum()} giá trị thiếu")

    # Xóa các giá trị ngoại lai
    if options.get("remove_outliers", False):
        numeric_cols = ["Potential", "Current"]
        z_scores = np.abs(zscore(processed_df[numeric_cols], nan_policy='omit'))
        mask = (z_scores < 3).all(axis=1)
        
        outlier_count = (~mask).sum()
        if outlier_count > 0:
            processed_df = processed_df[mask].reset_index(drop=True)
            print(f"Đã xóa {outlier_count} giá trị ngoại lai")
        else:
            print("Không tìm thấy giá trị ngoại lai")

    # Làm mịn tín hiệu
    if options.get("smooth", False):
        if len(processed_df) >= 5:
            window_length = min(51, len(processed_df) // 2 * 2 - 1)
            window_length = max(5, window_length)
            
            processed_df["Current"] = savgol_filter(
                processed_df["Current"].values,
                window_length,
                3
            )
            print(f"Đã áp dụng làm mịn với độ dài cửa sổ {window_length}")
        else:
            print("Không đủ điểm dữ liệu để làm mịn")

    # Hiệu chỉnh đường cơ bản
    if options.get("baseline_correction", False):
        processed_df["Current"] = detrend(processed_df["Current"].values)
        print("Đã áp dụng hiệu chỉnh đường cơ bản")

    return processed_df, transformers


def preprocess_wide_data(df, options=None):
    """
    Xử lý dữ liệu voltammetry dạng rộng (các cột điện thế) với các tùy chọn đã chỉ định.
    Hàm này làm việc trực tiếp trên dữ liệu dạng rộng gốc mà không chuyển đổi sang dạng dài.

    Args:
        df: DataFrame chứa các cột điện thế và cột metadata
        options: Từ điển chứa các tùy chọn xử lý

    Returns:
        DataFrame đã xử lý và từ điển các transformer
    """
    if options is None:
        options = {}

    processed_df = df.copy()
    transformers = {}

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
                        continue
            elif col not in columns_to_skip:
                metadata_columns.append(col)

    print(f"Found {len(voltage_columns)} voltage columns and {len(metadata_columns)} metadata columns")

    # Xử lý dữ liệu
    if voltage_columns:
        # Xử lý giá trị thiếu
        if options.get("fill_missing", False):
            imputer = SimpleImputer(strategy='mean')
            processed_df[voltage_columns] = imputer.fit_transform(processed_df[voltage_columns])
            transformers["impute"] = imputer

        # Xóa giá trị ngoại lai
        if options.get("remove_outliers", False):
            z_scores = np.abs(zscore(processed_df[voltage_columns], nan_policy='omit'))
            mask = (z_scores < 3).all(axis=1)
            outlier_count = (~mask).sum()
            if outlier_count > 0:
                processed_df = processed_df[mask].reset_index(drop=True)

        # Làm mịn tín hiệu
        if options.get("smooth", False):
            for col in voltage_columns:
                if len(processed_df) >= 5:
                    window_length = min(51, len(processed_df) // 2 * 2 - 1)
                    window_length = max(5, window_length)
                    processed_df[col] = savgol_filter(
                        processed_df[col].values,
                        window_length,
                        3
                    )

    return processed_df, transformers

    # Fill missing values
    if options.get("fill_missing", False):
        # Apply to voltage columns only
        for col in voltage_columns:
            if processed_df[col].isnull().any():
                # Fill missing values with column mean
                col_mean = processed_df[col].mean()
                processed_df[col].fillna(col_mean, inplace=True)
        print("Filled missing values in voltage columns")

    # Signal smoothing
    if options.get("smooth", False):
        # Apply smoothing to each voltage column
        for col in voltage_columns:
            # Ensure we have enough data points for smoothing
            if len(processed_df) >= 5:
                # Calculate appropriate window length (must be odd and less than data length)
                window_length = min(5, len(processed_df) // 2 * 2 - 1)

                # Ensure window length is at least 3 (minimum for Savitzky-Golay filter)
                window_length = max(3, window_length)

                # Apply Savitzky-Golay filter to the column
                processed_df[col] = savgol_filter(
                    processed_df[col].values,
                    window_length,
                    2  # Polynomial order
                )
        print(f"Applied smoothing to voltage columns with window length {window_length}")

    # Baseline correction
    if options.get("baseline_correction", False):
        # Apply detrending to each voltage column
        for col in voltage_columns:
            processed_df[col] = detrend(processed_df[col].values)
        print("Applied baseline correction to voltage columns")

    # Return the processed dataframe and transformers
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


def apply_wide_transformers(df, transformers):
    """
    Apply saved preprocessing transformers to new wide-format data.

    Args:
        df: DataFrame in wide format to transform
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

            # Apply transformer to all numeric columns
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()

            # Apply transformer
            transformed_data = transformer.transform(processed_df[numeric_cols])

            # Replace the original columns with transformed data
            processed_df[numeric_cols] = transformed_data

    return processed_df
