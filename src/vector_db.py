from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import os

COLLECTION_NAME = "financial_reports"
# Khởi tạo Qdrant lưu trữ dạng file cục bộ trên disk
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qdrant_client = QdrantClient(path=os.path.join(BASE_DIR, "data", "qdrant_storage"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def init_vector_database():
    # 1. Tạo cấu trúc Collection nếu chưa tồn tại
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE) # MiniLM trả về vector 384 dims
        )
    
    # 2. Quét thư mục market_reports đọc dữ liệu văn bản (Giả lập đơn giản cho file text/markdown)
    report_dir = os.path.join(BASE_DIR, "data", "market_reports")
    if os.path.exists(report_dir):
        points = []
        idx = 1
        for file_name in os.listdir(report_dir):
            if file_name.endswith(".txt") or file_name.endswith(".md"):
                with open(os.path.join(report_dir, file_name), "r", encoding="utf-8") as f:
                    text_content = f.read()
                    
                    # Naive Chunking đơn giản cho V1: Cắt mỗi 500 ký tự
                    chunks = [text_content[i:i+500] for i in range(0, len(text_content), 450)]
                    
                    for chunk in chunks:
                        vector = embedding_model.encode(chunk).tolist()
                        points.append(PointStruct(
                            id=idx,
                            vector=vector,
                            payload={"text": chunk, "source": file_name}
                        ))
                        idx += 1
                        
        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Đã nạp thành công {len(points)} chunks văn bản vào Qdrant DB.")