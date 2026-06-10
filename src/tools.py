# src/tools.py
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.database import query_sqlite, DB_PATH
from src.vector_db import qdrant_client, COLLECTION_NAME, embedding_model

def tool_run_sqlite_query(sql_command: str, db_path: str = "data/finance.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        rows = cursor.execute(sql_command).fetchall()
        clean_data = [dict(row) for row in rows]
        return clean_data, None 
    except Exception as e:
        return [], str(e)
    finally:
        conn.close()

def tool_semantic_rag_search(user_query: str) -> str:
    try:
        # 1. Embedding câu hỏi của user thành vector 384 chiều
        query_vector = embedding_model.encode(user_query).tolist()
        
        # 2. Phóng truy vấn tìm top 2 đoạn văn lân cận nhất trong Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=2
        )
        
        # 3. Gộp text kết quả trả về
        context_list = []
        for res in search_results:
            context_list.append(f"Content: {res.payload['text']} (Source: {res.payload['source']})")
            
        if not context_list:
            return "Không tìm thấy tài liệu phân tích vĩ mô liên quan trong Vector DB."
            
        return "\n\n".join(context_list)
    except Exception as e:
        return f"Lỗi truy vấn Vector DB: {str(e)}"
    
OUTPUT_CHART_PATH = "data/exports/market_chart.png"

def tool_generate_market_chart(ticker: str) -> str:
    if ticker == "UNKNOWN" or ticker == "NONE":
        return "Không có mã cụ thể để vẽ biểu đồ."
        
    try:
        import sqlite3
        conn = sqlite3.connect("data/finance_mvp.db")
        query = f"SELECT timestamp, price, volume FROM prices WHERE symbol = '{ticker}' ORDER BY timestamp ASC;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return f"Không có dữ liệu trong Database để vẽ biểu đồ cho mã {ticker}."
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['price'], marker='o', color='b', linestyle='-', linewidth=2, label='Giá trị (USD)')
        
        plt.title(f"BIỂU ĐỒ BIẾN ĐỘNG GIÁ TỰ ĐỘNG - THỰC THỂ: {ticker}", fontsize=14, fontweight='bold')
        plt.xlabel("Thời gian (Timestamp)", fontsize=10)
        plt.ylabel("Giá (USD)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(OUTPUT_CHART_PATH), exist_ok=True)
        plt.savefig(OUTPUT_CHART_PATH)
        plt.close() # Giải phóng bộ nhớ RAM đồ họa ngay lập tức để tránh rò rỉ bộ nhớ
        
        return f"Path: '{OUTPUT_CHART_PATH}'."
        
    except Exception as e:
        return f"Lỗi trong quá trình vẽ biểu đồ: {str(e)}"