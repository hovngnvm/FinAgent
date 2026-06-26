import os
import pandas as pd
from src.tools import tool_calculate_technical_indicators
from src.database import ingest_data_to_db, db_engine
from src.vector_db import ingest_data_to_qdrant, qdrant_client, embedding_model, COLLECTION_NAME
from src.chunking import advanced_parent_child_chunker

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def run_enterprise_spark_pipeline():
    os.makedirs("data/spark_market_checkpoints", exist_ok=True)
    os.makedirs("data/spark_news_checkpoints", exist_ok=True)

    spark = SparkSession.builder \
        .appName("FinAgent-Enterprise-Medallion-Processing") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

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

    # Luồng 1: Nhận diện biến động Giá đổ vào PostgreSQL
    market_kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "finagent_bronze_market") \
        .load()

    parsed_market_df = market_kafka_df \
        .selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), market_schema).alias("data")) \
        .select(col("data.ingest_timestamp").alias("timestamp"), "data.payload.*")

    def write_market_to_postgres(batch_df, batch_id):
        if batch_df.count() == 0: 
            return
            
        new_data_df = batch_df.toPandas()
        enriched_records = []
        unique_symbols = new_data_df['symbol'].unique()
    
        try:
            for symbol in unique_symbols:
                # Đọc dữ liệu lịch sử từ PostgreSQL Engine phục vụ tính toán cửa sổ trượt
                query = "SELECT * FROM prices WHERE symbol = %s ORDER BY timestamp DESC LIMIT 30"
                history_df = pd.read_sql(query, db_engine, params=(symbol,))
            
                current_symbol_df = new_data_df[new_data_df['symbol'] == symbol]
                
                if not history_df.empty:
                    combined_df = pd.concat([history_df, current_symbol_df], ignore_index=True)
                else:
                    combined_df = current_symbol_df
                
                calculated_df = tool_calculate_technical_indicators(combined_df)
                new_calculated_rows = calculated_df.tail(len(current_symbol_df))
                enriched_records.append(new_calculated_rows)
        except Exception as e:
            print(f"-> Lỗi trong quá trình xử lý Enrichment Micro-batch: {str(e)}")
        
        if enriched_records:
            final_df = pd.concat(enriched_records, ignore_index=True)
            # Lưu dồn dữ liệu đã tính toán xong xuôi vào Postgres Gold Layer
            ingest_data_to_db(
                dataframe=final_df, 
                table_name="prices", 
                mode="append"
            )

    query_market = parsed_market_df.writeStream \
        .foreachBatch(write_market_to_postgres) \
        .option("checkpointLocation", "data/spark_market_checkpoints") \
        .start()

    # Luồng 2: Nhận diện Tin tức đổ vào Qdrant Vector DB
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
        all_structured_chunks = []
        
        for _, row in pandas_df.iterrows():
            full_text_content = f"Tiêu đề: {row['title']}\nTóm tắt: {row['summary']}"
            formatted_chunks = advanced_parent_child_chunker(
                text=full_text_content,
                source_link=row['link'],
                parent_size=1200, # Kích thước khối ngữ cảnh tối ưu cho prompt LLM
                child_size=250    # Kích thước khối vector tối ưu cho độ nhạy cosine search
            )
            
            for chunk in formatted_chunks:
                chunk["timestamp"] = row['ingest_timestamp']

            all_structured_chunks.extend(formatted_chunks)
            
        if all_structured_chunks:
            ingest_data_to_qdrant(chunks_data=all_structured_chunks)

    query_news = parsed_news_df.writeStream \
        .foreachBatch(write_news_to_qdrant) \
        .option("checkpointLocation", "data/spark_news_checkpoints") \
        .start()

    print("streaming")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run_enterprise_spark_pipeline()