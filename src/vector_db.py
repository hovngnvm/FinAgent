import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder

# Khởi tạo kết nối Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)

COLLECTION_NAME = "financial_reports"

# Tải các mô hình phục vụ cascade pipeline
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Phục vụ Dense Vector (384 chiều)
reranking_model = CrossEncoder("BAAI/bge-reranker-large")  # Mô hình Reranker chốt hạ precision

def init_vector_database():
    """Khởi tạo Collection hỗ trợ Hybrid Search (Dense + Sparse/BM25) trực tiếp trên Qdrant"""
    # Kiểm tra xem collection đã tồn tại chưa để tránh ghi đè cấu hình
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # 1. Cấu hình cho Vector ngữ nghĩa (Dense)
            vectors_config=VectorParams(
                size=384, 
                distance=Distance.COSINE
            ),
            # 2. BẬT TÍNH NĂNG MỚI: Cấu hình cho Vector từ khóa chính xác (Sparse / BM25)
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=True # Tối ưu RAM, ghi chỉ mục xuống đĩa HNSW giống tư duy DE cứng
                    )
                )
            }
        )
        print(f"-> [Qdrant Enterprise]: Đã khởi tạo thành công Collection Hybrid Search: {COLLECTION_NAME}")
    else:
        print(f"-> [Qdrant Enterprise]: Collection '{COLLECTION_NAME}' đã tồn tại.")

def _text_to_sparse_vector(text_content: str) -> dict:
    """
    Hàm băm từ khóa thô thành Sparse Vector định dạng Qdrant (Term Frequency).
    Qdrant v1.x hỗ trợ nhận diện trực tiếp hoặc thông qua ánh xạ ID Token.
    """
    words = text_content.lower().split()
    frequency = {}
    for word in words:
        # Loại bỏ ký tự đặc biệt cơ bản quanh từ
        clean_word = "".join(ch for ch in word if ch.isalnum())
        if clean_word:
            frequency[clean_word] = frequency.get(clean_word, 0.0) + 1.0
            
    # Chuyển đổi sang định dạng vị trí (indices) và giá trị trọng số (values)
    # Dùng hàm băm hash() để map chuỗi text thành ID số nguyên (indices) cho Sparse Vector
    indices = []
    values = []
    for word, count in frequency.items():
        # Đảm bảo index là số nguyên dương trong dải của Qdrant
        indices.append(abs(hash(word)) % 1000000) 
        values.append(float(count))
        
    return {"indices": indices, "values": values}

def ingest_data_to_qdrant(chunks_data: list[dict]):
    """Nạp dữ liệu hỗ trợ cấu trúc Parent-Child cấp cao vào Qdrant"""
    if not chunks_data:
        return
        
    points = []
    for idx, chunk in enumerate(chunks_data):
        text_block = chunk["text"] # Child text phục vụ tính toán khoảng cách vector
        
        dense_emb = embedding_model.encode(text_block).tolist()
        sparse_emb = _text_to_sparse_vector(text_block)
        
        points.append(
            PointStruct(
                id=abs(hash(chunk["source"] + chunk["chunk_hierarchy"] + str(idx))) % 10000000,
                vector={
                    "": dense_emb,
                    "text-sparse": sparse_emb
                },
                payload={
                    "text": text_block,
                    "parent_text": chunk.get("parent_text", text_block), # LƯU TRỮ TRỰC TIẾP PARENT TEXT VÀO PAYLOAD
                    "source": chunk["source"],
                    "timestamp": chunk.get("timestamp", ""),
                    "hierarchy": chunk["chunk_hierarchy"]
                }
            )
        )
        
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"-> [Qdrant Ingestion]: Đã nạp thành công {len(points)} Parent-Child chunks.")