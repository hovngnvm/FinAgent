import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "finance.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "raw.csv")

def init_relational_database():
    # Kiểm tra và tạo bảng từ file CSV
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        conn = sqlite3.connect(DB_PATH)
        # Ghi đè dữ liệu sạch vào bảng prices
        df.to_sql("prices", conn, if_exists="replace", index=False)
        conn.close()
        print("-> Đã đồng bộ dữ liệu CSV vào SQLite thành công.")
    else:
        print("-> Lỗi: Không tìm thấy file raw.csv!")

def query_sqlite(sql_query: str) -> str:
    """Hàm bổ trợ giúp Agent thực thi câu lệnh SQL và trả về text kết quả"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df.to_string()
    except Exception as e:
        return f"Error executing SQL: {str(e)}"