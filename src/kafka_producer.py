import asyncio
import json
import websockets
import feedparser
import aiohttp
from datetime import datetime
from aiokafka import AIOKafkaProducer
from vnstock import Vnstock # Sử dụng thư viện vnstock4 thật mảng chứng khoán Việt Nam

KAFKA_BROKER = "localhost:9092"
TOPIC_MARKET = "finagent_bronze_market"
TOPIC_NEWS = "finagent_bronze_news"

# Khởi tạo instance vnstock4 mảng chứng khoán
stock_client = Vnstock()

async def get_producer():
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    await producer.start()
    return producer

async def stream_binance_websocket(producer):
    """Kết nối WebSocket trực tiếp đến Binance Live Stream API"""
    streams = "btcusdt@trade/ethusdt@trade"
    socket_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    print("-> [Bronze Ingestion]: Đang mở kết nối WebSocket trực tiếp tới Binance Trade Stream...")
    
    while True:
        try:
            async with websockets.connect(socket_url) as ws:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    # Mapping chuẩn Data Contract
                    payload = {
                        "ingest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "data_source": "binance_websocket_live",
                        "asset_class": "CRYPTO",
                        "payload": {
                            "symbol": data['s'],              # Ví dụ: BTCUSDT
                            "price": float(data['p']),         # Giá khớp lệnh thật
                            "volume": float(data['q'])         # Khối lượng thật
                        }
                    }
                    #### load to Kafka
                    await producer.send_and_wait(TOPIC_MARKET, payload) 
                    
        except websockets.exceptions.ConnectionClosed:
            print("-> [Bronze Ingestion]: Mất kết nối Binance WS. Đang kích hoạt cơ chế Auto-reconnect...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Lỗi luồng Binance WS: {str(e)}")
            await asyncio.sleep(5)

async def stream_vnstock_polling(producer):
    """Sử dụng thư viện vnstock bốc giá Realtime của các mã chứng khoán SSI, VND"""
    print("-> [Bronze Ingestion]: Đang khởi động luồng Polling REST API qua vnstock...")
    symbols = [
            # Thép & Công nghệ
            "HPG", "FPT", 
            # Chứng khoán (Biến động mạnh)
            "VIX", "SSI", "VND", 
            # Ngân hàng (Thanh khoản cao)
            "SHB", "STB", "VPB", "MBB", "TCB",
            # Bất động sản / Trụ
            "VIC", "VHM"
]
    
    while True:
        try:
            for sym in symbols:
                # Gọi hàm bốc giá thời gian thực thật từ vnstock
                stock_data = stock_client.stock(symbol=sym, source='VCI').trading.price_board()
                if not stock_data.empty:
                    latest_row = stock_data.iloc[0]
                    payload = {
                        "ingest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "data_source": "vnstock_api_vci",
                        "asset_class": "STOCK",
                        "payload": {
                            "symbol": sym,
                            "price": float(latest_row['matchPrice']), # Giá khớp thật sàn VN
                            "volume": float(latest_row['matchVolume']) # Khối lượng khớp thật
                        }
                    }
                    await producer.send_and_wait(TOPIC_MARKET, payload)
            await asyncio.sleep(2) # Polling rate 2 giây/lần đạt chuẩn REST Rate-limit
        except Exception as e:
            print(f"Lỗi luồng vnstock: {str(e)}")
            await asyncio.sleep(5)

async def stream_rss_news_feeds(producer):
    """Cào trực tiếp tin tức kinh tế vĩ mô từ các kênh RSS lớn (Cafef / VnExpress)"""
    print("-> [Bronze Ingestion]: Khởi động luồng lắng nghe tin tức RSS Feeds thật...")
    rss_urls = [
        "https://cafef.vn/thi-truong-chung-khoan.rss",
        "https://cafef.vn/kinh-te-vi-mo.rss",
        "https://cafef.vn/doanh-nghiep.rss",
        "https://vietstock.vn/rss/tin-moi-nhat.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss"
    ]
    seen_guid = set() # Bộ nhớ đệm chống trùng lặp tin bài cũ
    
    while True:
        try:
            for url in rss_urls:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]: # Lấy 3 tin mới nhất mỗi chu kỳ quét
                    guid = entry.get('id', entry.get('link', ''))
                    if guid not in seen_guid:
                        seen_guid.add(guid)
                        
                        payload = {
                            "ingest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "data_source": "rss_financial_feed",
                            "payload": {
                                "title": entry.get('title', ''),
                                "summary": entry.get('summary', entry.get('description', '')),
                                "link": entry.get('link', ''),
                                "published": entry.get('published', datetime.now().strftime("%Y-%m-%d"))
                            }
                        }
                        await producer.send_and_wait(TOPIC_NEWS, payload)
            await asyncio.sleep(15) # Quét RSS định kỳ 15 giây/lần
        except Exception as e:
            print(f"Lỗi luồng RSS News: {str(e)}")
            await asyncio.sleep(10)

async def main():
    producer = await get_producer()
    try:
        # Fan-in Asyncio Gather: Phóng song song 4 luồng dữ liệu thật chạy bất đồng bộ đồng thời
        await asyncio.gather(
            stream_binance_websocket(producer),
            stream_vnstock_polling(producer),
            stream_rss_news_feeds(producer)
        )
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(main())