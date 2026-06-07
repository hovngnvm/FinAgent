# src/tools.py
import sqlite3
from src.database import query_sqlite, DB_PATH
from src.vector_db import qdrant_client, COLLECTION_NAME, embedding_model

def tool_read_market_database(ticker: str) -> str:
    """Tool chui vào SQLite bốc dữ liệu lịch sử giá của một mã cụ thể"""
    if ticker == "UNKNOWN" or ticker == "NONE":
        return "Hệ thống không tìm thấy mã chứng khoán/crypto cụ thể trong câu hỏi để truy vấn SQL."
    
    # Viết câu lệnh SQL cứng để bảo vệ DB chống Prompt Injection SQL
    sql_command = f"SELECT * FROM prices WHERE symbol = '{ticker}' ORDER BY timestamp DESC LIMIT 5;"             ### CÓ THỂ SỬ DỤNG AGENT ĐỂ VIẾT SQL KHÔNG?
    return query_sqlite(sql_command)

def tool_semantic_rag_search(user_query: str) -> str:
    """Tool bắn vector search xuống Qdrant lấy các đoạn văn tri thức vĩ mô"""
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