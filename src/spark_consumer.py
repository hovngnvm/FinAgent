from PRJCT.CRYPTO_STREAMING.scripts.consumer import spark_consumer
import os, sqlite3, pandas as pd
from src.tools import tool_calculate_technical_indicators
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def run_enterprise_spark_pipeline():
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 src/spark_analytics.py'
    
    spark = SparkSession.builder \
        .appName("FinAgent-Enterprise-Medallion-Processing") \
        .config("spark.sql.shuffle.partitions", "2") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # ==========================================
    # CẤU HÌNH SCHEMA CHO 2 LUỒNG DỮ LIỆU THẬT
    # ==========================================
    market_schema = StructType([
        StructField("ingest_timestamp", StringType(), True),
        StructField("data_source", StringType(), True),
        StructField("payload", StructType([
            StructField("symbol", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("volume", DoubleType(), True)
        ]), True)
    ])

    news_schema = StructType([
        StructField("ingest_timestamp", StringType(), True),
        StructField("data_source", StringType(), True),
        StructField("payload", StructType([
            StructField("title", StringType(), True),
            StructField("summary", StringType(), True),
            StructField("link", StringType(), True)
        ]), True)
    ])

    # ==========================================
    # LUỒNG 1: XỬ LÝ CHUỖI THỜI GIAN GIÁ CỨNG ĐỔ VÀO SQLITE
    # ==========================================
    market_kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "finagent_bronze_market") \
        .load()

    parsed_market_df = market_kafka_df \
        .selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), market_schema).alias("data")) \
        .select("data.ingest_timestamp", "data.payload.*")

    def write_market_to_sqlite(batch_df, batch_id):
        from src.database import ingest_data_to_sqlite, DB_PATH
        if batch_df.count() == 0: 
            return
            
        # Bước 1: Chuyển dữ liệu mới của batch này sang Pandas (Chỉ có vài dòng)
        new_data_df = batch_df.toPandas()
        
        # Bước 2: Truy cập vào SQLite lấy thêm dữ liệu quá khứ để làm giàu (Enrichment)
        conn = sqlite3.connect(DB_PATH)
    
        enriched_records = []
        # Lấy danh sách các mã xuất hiện trong batch này (ví dụ: ['HPG', 'FPT'])
        unique_symbols = new_data_df['symbol'].unique()
    
        for symbol in unique_symbols:
            # Lấy tối đa 30 bản ghi gần nhất của mã này từ DB để có đủ cửa sổ trượt (window) tính toán
            query = f"SELECT * FROM prices WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 30"
            history_df = pd.read_sql(query, conn)
        
            # Gom dữ liệu mới của mã này trong batch
            current_symbol_df = new_data_df[new_data_df['symbol'] == symbol]
            
            # Gộp lịch sử và hiện tại lại với nhau
            combined_df = pd.concat([history_df, current_symbol_df], ignore_index=True)
            
            # Tính toán indicator trên tập dữ liệu đã có quá khứ (Đảm bảo tính ĐÚNG)
            calculated_df = tool_calculate_technical_indicators(combined_df)
            
            # Chỉ lấy lại những dòng mới (những dòng thuộc về batch hiện tại) để chuẩn bị ghi dồn vào DB
            # Tránh ghi đè trùng lặp dữ liệu lịch sử cũ
            new_calculated_rows = calculated_df.tail(len(current_symbol_df))
            enriched_records.append(new_calculated_rows)
        
        conn.close()
        
        # Bước 3: Gộp tất cả các mã lại và nạp dồn vào SQLite
        if enriched_records:
            final_df = pd.concat(enriched_records, ignore_index=True)
            
            ingest_data_to_sqlite(
                db_path=DB_PATH, 
                dataframe=final_df, 
                table_name="prices", 
                mode="append"
            )

    query_market = parsed_market_df.writeStream \
        .foreachBatch(write_market_to_sqlite) \
        .option("checkpointLocation", "data/spark_market_checkpoints") \
        .start()

    # ==========================================
    # LUỒNG 2: REAL-TIME INGESTION TEXT TIN TỨC THẬT ĐỔ VÀO QDRANT VECTOR DB
    # ==========================================
    news_kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "finagent_bronze_news") \
        .load()

    parsed_news_df = news_kafka_df \
        .selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), news_schema).alias("data")) \
        .select("data.ingest_timestamp", "data.payload.*")

    def write_news_to_qdrant(batch_df, batch_id):
        if batch_df.count() == 0: return
        pandas_df = batch_df.toPandas()
        
        from src.vector_db import ingest_data_to_qdrant, qdrant_client, embedding_model, COLLECTION_NAME
            
        prepared_chunks = []
        for _, row in pandas_df.iterrows():
            text_block = f"Tiêu đề: {row['title']}\nTóm tắt: {row['summary']}"
            
            prepared_chunks.append({
                "text": text_block,
                "source": row['link'],
                "timestamp": row['ingest_timestamp']
            })
        
        ingest_data_to_qdrant(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            COLLECTION_NAME=COLLECTION_NAME,
            chunks_data=prepared_chunks
        )

    query_news = parsed_news_df.writeStream \
        .foreachBatch(write_news_to_qdrant) \
        .option("checkpointLocation", "data/spark_news_checkpoints") \
        .start()

    print("-> [Hạ tầng Phân tán]: Cả hai đường ống Stream (SQLite & Qdrant DB) từ nguồn thật đã kích hoạt!")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run_enterprise_spark_pipeline()