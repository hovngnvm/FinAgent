# main.py
import os
from dotenv import load_dotenv
load_dotenv

from src.database import init_relational_database, DB_PATH
from src.vector_db import init_vector_database, qdrant_client, COLLECTION_NAME
from src.agent_graph import app
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

def main():
    print("=== FINAGENT V1 ===")
    
    init_relational_database(DB_PATH)
    init_vector_database(qdrant_client, COLLECTION_NAME)
    
    langfuse_handler = CallbackHandler()
    
    chat_config = {"configurable": {"thread_id": "finance_session"},
                   "callbacks": [langfuse_handler]}
    
    while True:
        user_query = input("\nUser (Gõ 'exit' để thoát): ")
        if user_query.lower() == 'exit':
            break
            
        # ⚡ ĐIỂM ĐẮT GIÁ: Không khởi tạo đè NONE cho current_target nữa!
        # Ta chỉ đút tin nhắn mới nhất vào lịch sử, LangGraph Checkpointer sẽ tự động 
        # lôi mảng messages và target cũ của thread_id này dưới RAM lên gộp lại!
        state_update = {
            "messages": [HumanMessage(content=user_query)]
        }
        
        # Phóng đồ thị chạy kèm thẻ định danh phiên chat chat_config
        final_state = app.invoke(state_update, config=chat_config)
        
        last_msg = final_state["messages"][-1]
        print(f"\nFinAgent: {last_msg.content}")
        print("-" * 50)

if __name__ == "__main__":
    main()