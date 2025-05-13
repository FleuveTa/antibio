import pandas as pd

def create_clean_csv(input_file, output_file, rows_to_remove=None):
    """
    Đọc file CSV input, xóa các dòng được chỉ định, và lưu thành file CSV mới
    
    Args:
        input_file: Đường dẫn đến file CSV đầu vào
        output_file: Đường dẫn đến file CSV đầu ra
        rows_to_remove: Danh sách các chỉ số dòng cần xóa (bắt đầu từ 0)
    """
    try:
        # Đọc file CSV
        print(f"Đang đọc file {input_file}...")
        df = pd.read_csv(input_file)
        
        original_rows = len(df)
        print(f"File gốc có {original_rows} dòng.")
        
        # Xóa các dòng được chỉ định
        if rows_to_remove:
            print(f"Xóa các dòng với chỉ số: {rows_to_remove}")
            df = df.drop(rows_to_remove)
        
        # Đặt lại chỉ số
        df = df.reset_index(drop=True)
        
        # Kiểm tra giá trị NaN
        na_count = df.isna().sum().sum()
        if na_count > 0:
            print(f"Cảnh báo: Có {na_count} giá trị NaN trong dữ liệu.")
            print("Số lượng NaN theo cột:")
            print(df.isna().sum())
            
            # Tùy chọn: Thay thế giá trị NaN
            # df = df.fillna(0)  # Thay thế NaN bằng 0
        
        # Lưu thành file mới
        df.to_csv(output_file, index=False)
        print(f"Đã lưu file mới với {len(df)} dòng tại {output_file}")
        
        return True
    except Exception as e:
        print(f"Lỗi: {e}")
        return False

# Sử dụng hàm
if __name__ == "__main__":
    # Thay đổi đường dẫn file đầu vào và đầu ra tại đây
    input_csv = "D:/thesis/antibioticas/python_app_builder/data/raw\experiment_20250416_215159/all_data_IviumData.csv"  # Đổi thành đường dẫn file thực tế của bạn
    output_csv = "all_data_invium_cut3.csv"  # Tên file mới sau khi xóa dòng
    
    # Chỉ định các dòng cần xóa: 13, 82, 238
    rows_to_remove = [13, 82, 238]
    
    create_clean_csv(input_csv, output_csv, rows_to_remove)
