import os
import json
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from psycopg2.extras import RealDictCursor
from sqlalchemy import text
from langchain_ollama import OllamaLLM
from qdrant_client.models import PrefetchQuery, QueryRequest

from src.database import db_engine, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from src.vector_db import qdrant_client, COLLECTION_NAME, embedding_model, reranking_model, _text_to_sparse_vector

# Khởi tạo instance LLM phụ trợ riêng cho các tác vụ Context Engineering
rag_llm = OllamaLLM(model="qwen2.5-coder:3b-instruct-q5_K_S", temperature=0.1)

def tool_run_sqlite_query(sql_command: str):
    """Driver kết nối PostgreSQL phục vụ SQL Agent truy vấn dữ liệu lớn (Đã đồng bộ ở GĐ1)"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql_command)
        if cursor.description:
            rows = cursor.fetchall()
            return [dict(row) for row in rows], None
        conn.commit()
        return [{"status": "Success"}], None
    except Exception as e:
        return [], str(e)
    finally:
        if conn:
            conn.close()

def tool_semantic_rag_search(processed_query: str) -> str:
    try:
        dense_vector = embedding_model.encode(processed_query).tolist()
        sparse_vector = _text_to_sparse_vector(processed_query)
        
        prefetch_dense = PrefetchQuery(vector=dense_vector, limit=10)
        prefetch_sparse = PrefetchQuery(vector={"name": "text-sparse", "vector": sparse_vector}, limit=10)
        
        hybrid_results = qdrant_client.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=[QueryRequest(prefetch=[prefetch_dense, prefetch_sparse], limit=10)]
        )
        search_results = hybrid_results[0].points if hybrid_results else []
        
        if not search_results:
            return "Không tìm thấy tài liệu phân tích vĩ mô liên quan trong Vector DB."
            
        # CHẤM ĐIỂM NGỮ NGHĨA DỰA TRÊN CHILD CHUNK ĐỂ ĐẢM BẢO PRECISION ĐẠT ĐỈNH
        rerank_pairs = [(processed_query, res.payload.get('text', '')) for res in search_results]
        scores = reranking_model.predict(rerank_pairs)
        
        ranked_docs = []
        for idx, score in enumerate(scores):
            ranked_docs.append({
                # ĐỘT PHÁ KIẾN TRÚC: Bốc Parent Text lớn thay vì Child Text thô
                "parent_context": search_results[idx].payload.get('parent_text', search_results[idx].payload.get('text', '')),
                "source": search_results[idx].payload.get('source', 'N/A'),
                "rerank_score": float(score)
            })
        ranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        seen_contexts = set()
        final_context_list = []
        
        for doc in ranked_docs:
            if doc["rerank_score"] < 0.1: 
                continue
            # Lọc trùng lặp trên tầng Parent Context lớn
            context_hash = doc["parent_context"][:150]
            if context_hash not in seen_contexts:
                seen_contexts.add(context_hash)
                final_context_list.append(
                    f"Context: {doc['parent_context']} (Source: {doc['source']} | Score: {doc['rerank_score']:.2f})"
                )
            if len(final_context_list) >= 2: # Giới hạn 2 Parent lớn để tránh làm loãng Token Context
                break

        return "\n\n".join(final_context_list)
    except Exception as e:
        return f"Lỗi trong hệ thống Hybrid Parent-Child RAG: {str(e)}"
    
OUTPUT_CHART_PATH = "data/exports/market_chart.png"

def tool_generate_market_chart(ticker: str) -> str:
    """Kết xuất đồ thị giá tự động thích ứng với cấu trúc PostgreSQL Gold Layer"""
    if not ticker or ticker in ["UNKNOWN", "NONE"]:
        return "Không có mã cụ thể để vẽ biểu đồ."
        
    try:
        query = "SELECT timestamp, price, volume FROM prices WHERE symbol = %s ORDER BY timestamp ASC;"
        df = pd.read_sql_query(query, db_engine, params=(ticker,))
        
        if df.empty:
            return f"Không có dữ liệu trong Database để vẽ biểu đồ cho mã {ticker}."
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
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
        return f"Path: '{OUTPUT_CHART_PATH}'."
    except Exception as e:
        return f"Lỗi trong quá trình vẽ biểu đồ: {str(e)}"
    finally:
        plt.close('all')

def tool_calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Mô-đun Data Engineering: Tính toán chuỗi chỉ số kỹ thuật sliding window"""
    if df.empty: 
        return df
        
    df = df.copy()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'symbol', 'price'])
    
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    df['SMA_5'] = df.groupby('symbol')['price'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['SMA_20'] = df.groupby('symbol')['price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    def calc_rsi(series):
        if len(series) < 14: 
            return 50.0
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('symbol')['price'].transform(calc_rsi)
    
    df['MACD'] = df.groupby('symbol')['price'].transform(lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean())
    df['MACD_Signal'] = df.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    df = df.bfill().ffill()
    return df