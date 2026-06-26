import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Tạo URI kết nối PostgreSQL từ biến môi trường
DB_USER = os.getenv("DB_USER", "finuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "finpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "finance_db")

# Dùng kết nối trực tiếp nội bộ hoặc từ host ngoài dựa trên ngữ cảnh chạy
POSTGRES_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db_engine = create_engine(POSTGRES_URI)

def init_relational_database():
    """Khởi tạo kết nối và tạo cấu trúc bảng tối ưu trên PostgreSQL"""
    with db_engine.begin() as conn:
        # Tạo bảng giá chuẩn hóa dữ liệu định lượng
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                timestamp TIMESTAMP,
                symbol VARCHAR(20),
                price DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                SMA_5 DOUBLE PRECISION,
                SMA_20 DOUBLE PRECISION,
                RSI DOUBLE PRECISION,
                MACD DOUBLE PRECISION,
                MACD_Signal DOUBLE PRECISION
            );
        """))
        
        # Thiết lập Index tối ưu hóa tốc độ tìm kiếm Time-Series cho SQL Agent
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON prices (symbol, timestamp DESC);"))
    print(f"-> [PostgreSQL]: Đã khởi tạo cấu trúc cơ sở dữ liệu Enterprise thành công tại {DB_HOST}:{DB_PORT}")

def ingest_data_to_db(dataframe, table_name="prices", mode="append"):
    """
    Nhận vào một Pandas DataFrame và nạp dồn vào PostgreSQL.
    Tên hàm được giữ nguyên để tránh làm gãy các dependency import khác.
    """
    if dataframe is None or dataframe.empty:
        return
        
    # Chuyển đổi timestamp sang đúng định dạng trước khi đổ vào Postgres TIMESTAMP
    if 'timestamp' in dataframe.columns:
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
        
    if_exists_mode = "replace" if mode == "replace" else "append"
    
    # Ghi dữ liệu hiệu năng cao thông qua SQLAlchemy Engine
    dataframe.to_sql(table_name, con=db_engine, if_exists=if_exists_mode, index=False, method='multi')