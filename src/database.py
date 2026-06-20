from langfuse.api.scim.types import scim_feature_support
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "finance.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "raw.csv")

def init_relational_database(db_path):
    """Khởi tạo kết nối và tạo index tối ưu cho bảng prices nếu chưa có"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng trống trước nếu chưa tồn tại (để tránh lỗi khi tạo index trên bảng không tồn tại)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            timestamp TEXT,
            symbol TEXT,
            price REAL,
            volume REAL,
            SMA_5 REAL,
            SMA_20 REAL,
            RSI REAL,
            MACD REAL,
            MACD_Signal REAL
        )
    """)
    
    # Tạo index tối ưu cho việc truy vấn mã chứng khoán theo thời gian mới nhất
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON prices (symbol, timestamp DESC);")
    
    conn.commit()
    conn.close()
    print(f"-> Đã khởi tạo cấu trúc SQLite và Index thành công tại: {db_path}")

def ingest_data_to_sqlite(db_path, dataframe, table_name="prices", mode="append"):
    """
    Nhận vào một Pandas DataFrame và nạp vào SQLite.
    - mode="append": Thêm tiếp dữ liệu vào (Tối ưu cho Streaming / Nạp dồn).
    - mode="replace": Xóa bảng cũ tạo lại bảng mới (Tối ưu cho thiết lập lại từ đầu).
    """
    if dataframe is None or dataframe.empty:
        print("-> Không có dữ liệu để nạp vào SQLite.")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Ghi dữ liệu vào bảng
    dataframe.to_sql(table_name, conn, if_exists=mode, index=False)
    
    conn.close()