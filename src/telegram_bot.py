import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import ứng dụng đồ thị đã compile cùng bộ nhớ Redis từ Giai đoạn trước
from src.agent_graph import app

load_dotenv()

# Cấu hình logging hệ thống để dễ dàng debug lỗi token hay kết nối mạng
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHART_FILE_PATH = "data/exports/market_chart.png"

async def command_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /start - Chào mừng người dùng gia nhập hệ thống phân tích"""
    welcome_text = (
        "🏦 *Chào mừng bạn đến với Enterprise FinAgent Platform!*\n\n"
        "Tôi là một Hệ thống Đa tác tử (Multi-Agent) thông minh kết nối hạ tầng dữ liệu thời gian thực "
        "PostgreSQL và Qdrant Vector DB.\n\n"
        "📊 *Bạn có thể ra lệnh cho tôi dưới dạng câu thoại tự nhiên hoặc cấu trúc sau:*\n"
        "• `/analyze HPG` - Phân tích chuyên sâu mã cổ phiếu/crypto\n"
        "• Hoặc chat trực tiếp: _'Mã BTC hôm nay có biến động gì không và vẽ đồ thị giúp tôi'_"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def process_agent_workflow(user_message: str, chat_id: int) -> str:
    """
    Hàm bao đóng (Wrapper) chạy luồng đồ thị LangGraph Sync 
    bên trong môi trường Async của Telegram.
    """
    # Khởi tạo config bốc thread_id dựa trên định danh duy nhất của Chat ID người dùng
    config = {"configurable": {"thread_id": f"telegram_user_{chat_id}"}}
    inputs = {"messages": [HumanMessage(content=user_message)]}
    
    # Chạy đồ thị chặn (Sync block) trong Thread Pool biệt lập của Asyncio để tránh treo Event Loop
    def run_graph():
        # Thực thi đồ thị và bốc bản ghi cuối cùng (Assistant Response)
        result = app.invoke(inputs, config=config)
        return result

    loop = asyncio.get_running_loop()
    final_state = await loop.run_in_executor(None, run_graph)
    
    # Trích xuất câu trả lời text cuối cùng của Agent Analyst
    if final_state and "messages" in final_state and final_state["messages"]:
        return final_state["messages"][-1].content
    return "Hệ thống tác tử không phản hồi. Vui lòng thử lại sau."

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bộ xử lý tin nhắn thoại tự nhiên từ người dùng"""
    chat_id = update.effective_chat.id
    user_text = update.message.text
    
    # Gửi hiệu ứng "typing..." tạo cảm giác AI đang xử lý thời gian thực chuyên nghiệp
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        # Xóa file biểu đồ cũ nếu có để tránh gửi nhầm đồ thị của phiên phân tích trước
        if os.path.exists(CHART_FILE_PATH):
            os.remove(CHART_FILE_PATH)
            
        # Gọi mạng lưới Multi-Agent xử lý logic (PostgreSQL + RAG + Chart Generation)
        ai_response = await process_agent_workflow(user_text, chat_id)
        
        # Gửi kết quả phân tích bằng văn bản cho người dùng
        await update.message.reply_text(ai_response)
        
        # ĐỒNG BỘ ĐỒ THỊ: Nếu Chart_Agent vừa vẽ xong đồ thị mới thành công xuống đĩa vật lý
        if os.path.exists(CHART_FILE_PATH):
            await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
            with open(CHART_FILE_PATH, "rb") as chart_img:
                await update.message.reply_photo(
                    photo=chart_img,
                    caption=f"📈 Biểu đồ dữ liệu thời gian thực được kết xuất tự động cho phiên truy vấn vừa rồi."
                )
                
    except Exception as e:
        logger.error(f"Lỗi xử lý tin nhắn: {str(e)}")
        await update.message.reply_text(f"❌ Có lỗi xảy ra trong quá trình đối soát dữ liệu: {str(e)}")

async def handle_analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bộ xử lý lệnh /analyze ngắn gọn cho dân DE/AIE chuyên nghiệp"""
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("⚠️ Vui lòng nhập kèm mã tài sản. Ví dụ: `/analyze BTC`", parse_mode="Markdown")
        return
        
    ticker = context.args[0].upper()
    simulated_message = f"Phân tích chuyên sâu mã {ticker} và vẽ biểu đồ biến động giá của nó."
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        if os.path.exists(CHART_FILE_PATH):
            os.remove(CHART_FILE_PATH)
            
        ai_response = await process_agent_workflow(simulated_message, chat_id)
        await update.message.reply_text(ai_response)
        
        if os.path.exists(CHART_FILE_PATH):
            with open(CHART_FILE_PATH, "rb") as chart_img:
                await update.message.reply_photo(photo=chart_img, caption=f"📈 Biểu đồ phân tích kỹ thuật: {ticker}")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")

def main():
    """Khởi động Polling Engine cho Telegram Bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ THIẾT LẬP THẤT BẠI: Thiếu biến TELEGRAM_BOT_TOKEN trong file .env")
        return

    # Khởi tạo ứng dụng Application nền tảng Async mới của python-telegram-bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Đăng ký các bộ định tuyến lệnh (Command & Message Routers)
    application.add_handler(CommandHandler("start", command_start))
    application.add_handler(CommandHandler("analyze", handle_analyze_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    print(f"-> [Telegram Automation]: Bot AI đang chạy và lắng nghe tin nhắn trên Polling mode...")
    application.run_polling()