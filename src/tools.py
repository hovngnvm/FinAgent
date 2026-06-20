# src/tools.py
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.database import DB_PATH
from src.vector_db import qdrant_client, COLLECTION_NAME, embedding_model, reranking_model

def tool_run_sqlite_query(sql_command: str, db_path: str = DB_PATH):
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
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        
        if not search_results:
            return "Không tìm thấy tài liệu phân tích vĩ mô liên quan trong Vector DB."
        
        rerank_pairs = []
        for res in search_results:
            rerank_pairs.append((user_query, res.payload['text']))
        
        scores = reranking_model.predict(rerank_pairs)
        
        ranked_docs = []
        for idx, score in enumerate(scores):
            ranked_docs.append({
                "text": search_results[idx].payload['text'],
                "source": search_results[idx].payload['source'],
                "rerank_score": float(score)
            })
        
        ranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        top_k_docs = ranked_docs[:2]
        
        context_list = []
        for doc in top_k_docs:
            context_list.append(f"Content: {doc['text']} (Source: {doc['source']} | Relevance: {doc['rerank_score']:.2f})")

        return "\n\n".join(context_list)
    
    except Exception as e:
        return f"Lỗi truy vấn Vector DB: {str(e)}"
    
OUTPUT_CHART_PATH = "data/exports/market_chart.png"

def tool_generate_market_chart(ticker: str) -> str:
    if ticker == "UNKNOWN" or ticker == "NONE":
        return "Không có mã cụ thể để vẽ biểu đồ."
        
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
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

def tool_calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán chỉ số chính xác bằng cách GROUPBY theo từng mã cổ phiếu
    và yêu cầu DataFrame đầu vào phải có đủ dữ liệu lịch sử.
    """
    if df.empty: 
        return df
        
    df['price'] = df['price'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # 1. SMA - Tính toán biệt lập theo từng mã bằng groupby
    df['SMA_5'] = df.groupby('symbol')['price'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['SMA_20'] = df.groupby('symbol')['price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # 2. RSI - Tính toán biệt lập theo từng mã bằng groupby
    def calc_rsi(series):
        if len(series) < 14: 
            return 50.0
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('symbol')['price'].transform(calc_rsi)
    
    # 3. MACD
    df['MACD'] = df.groupby('symbol')['price'].transform(lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean())
    df['MACD_Signal'] = df.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    df = df.bfill().ffill()
    df['timestamp'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df