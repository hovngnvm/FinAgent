# main.py
from src.database import init_relational_database
from src.vector_db import init_vector_database
from src.agent_graph import app
from langchain_core.messages import HumanMessage

def main():
    print("=== FINAGENT V1 ===")
    
    init_relational_database()
    init_vector_database()
    
    print("\nHệ thống sẵn sàng! Hãy nhập câu hỏi thử nghiệm.")
    user_input = "Cho tôi biết giá đóng cửa gần đây nhất của mã BTC."
    print(user_input)
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_target": "NONE"
    }
    
    # Phóng luồng chạy qua Đồ thị LangGraph
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("📊 (FINAL STATE):")
    print(f"Target Asset (Thực thể phân tích): {final_state.get('current_target')}")
    print("Lịch sử hội thoại & thực thi Tool:")
    for msg in final_state.get("messages", []):
        role = "Unknown"
        if isinstance(msg, dict):
            role = msg.get("role", "Assistant").capitalize()
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif msg.__class__.__name__ == "ToolMessage":
            role = f"Tool ({msg.name})"
        elif hasattr(msg, "type"):
            role = msg.type.capitalize()
            
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        print(f"  [{role}]:\n{content}\n")
    print("="*50)
    
    last_msg = final_state["messages"][-1]
    
    print(f"\nFinAgent:\n{last_msg.get('content') if isinstance(last_msg, dict) else last_msg.content}")

if __name__ == "__main__":
    main()