from IPython.core import payload
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer, CrossEncoder
import os, uuid

COLLECTION_NAME = "financial_reports"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

qdrant_client = QdrantClient(path=os.path.join(BASE_DIR, "data", "qdrant_storage"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranking_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

def init_vector_database(qdrant_client, COLLECTION_NAME):
    """Khởi tạo cấu trúc Collection trong Qdrant nếu chưa tồn tại"""
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # MiniLM trả về vector 384 dims, dùng khoảng cách Cosine để tìm kiếm độ tương đồng
            vectors_config=VectorParams(size=384, distance=Distance.COSINE) 
        )
        print(f"Đã tạo thành công Collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} đã tồn tại.")

# def scan_and_chunks(base_dir):
#     """Quét thư mục market_reports và cắt nhỏ văn bản thành các chunks"""
#     report_dir = os.path.join(base_dir, "data", "market_reports")
#     all_chunks = []
    
#     if not os.path.exists(report_dir):
#         print(f"Thư mục {report_dir} không tồn tại.")
#         return all_chunks

#     for file_name in os.listdir(report_dir):
#         if file_name.endswith(".txt") or file_name.endswith(".md"):
#             with open(os.path.join(report_dir, file_name), "r", encoding="utf-8") as f:
#                 text_content = f.read()
                
#                 # Thực hiện Chunking (Cắt mỗi 500 ký tự, gối đầu 50 ký tự)
#                 chunks = [text_content[i:i+500] for i in range(0, len(text_content), 450)]
                
#                 for chunk in chunks:
#                     all_chunks.append({
#                         "text": chunk,
#                         "source": file_name
#                     })
#     return all_chunks

def ingest_data_to_qdrant(qdrant_client, embedding_model, COLLECTION_NAME, chunks_data):
    """Nhận vào danh sách dữ liệu thô, tạo embedding và đẩy vào Qdrant"""
    if not chunks_data:
        return

    points = []
    # Sử dụng hàm generate_id() của qdrant hoặc dùng uuid/đếm số để làm ID cho Point
    for item in chunks_data:
        # Tạo vector embedding từ mô hình
        vector = embedding_model.encode(item["text"]).tolist()
        
        payload = {k: v for k, v in item.items() if k != "text"}
        payload["text"] = item["text"]
        
        point_id = item.get("id", str(uuid.uuid4()))
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        ))
        
    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)