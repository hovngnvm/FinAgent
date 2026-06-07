from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
import json

from src.tools import tool_read_market_database, tool_semantic_rag_search

class AgentState(TypedDict):
    # Lưu vết toàn bộ chuỗi lịch sử hội thoại/gọi tool
    messages: Annotated[Sequence[AnyMessage], add_messages]
    # Bộ nhớ ghi nhớ thực thể đang phân tích (Ví dụ: "BTC")
    current_target: str

llm = OllamaLLM(model="qwen2.5-coder:7b-instruct-q5_K_S", temperature=0, format="json")


def node_driver(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content # latest request from state
    
    # Kỹ thuật System Prompt Ép Kiểu Đầu Ra (Structured JSON Parser) giúp bóc tách thực thể
    system_instruction = (
        "You are the conductor of the financial system FinAgent.\n"
        "Your task is to read the user's question and extract which investment code they are referring to "
        "(For example: BTC, ETH, AAPL). If found, output in JSON format: {\"target\": \"MÃ\"}.\n"
        "If not found, leave: {\"target\": \"UNKNOWN\"}.\n"
        "Always respond in Vietnamese."
    )
    
    # Gọi model nhẩm từ
    response = llm.invoke(f"{system_instruction}\n\nUser Question: {last_message}")
    
    # Trích xuất target để cập nhật vào Ô nhớ State của Đồ thị một cách tự động
    try:
        parsed_json = json.loads(response) #json mode
        target = parsed_json.get("target", "UNKNOWN")
    except:
        target = "UNKNOWN"
        
    # Trả về bản cập nhật State. LangGraph sẽ tự gộp biến current_target này lại.
    return {"current_target": target}

def node_execute_sql(state: AgentState):
    ticker = state["current_target"]
    # Gọi hàm python thực tế dưới đĩa cứng
    sql_data = tool_read_market_database(ticker)
    
    # Đóng gói kết quả thành một ToolMessage để lưu lại vào lịch sử đồ thị
    return {"messages": [ToolMessage(content=sql_data, name="sqlite_query", tool_call_id="sqlite_query_call")]}

def node_execute_rag(state: AgentState):
    last_user_message = state["messages"][0].content
    rag_context = tool_semantic_rag_search(last_user_message)
    
    return {"messages": [ToolMessage(content=rag_context, name="qdrant_rag", tool_call_id="qdrant_rag_call")]}

def node_final_analyst(state: AgentState):
    # Khởi tạo lại một con LLM thường (tắt JSON mode) để nó viết văn bản tự nhiên cho mượt
    analyst_llm = OllamaLLM(model="qwen3:8b-q4_K_M", temperature=0.3)
    
    messages = state["messages"]
    target = state["current_target"]
    
    system_prompt = (
        "You are a Senior Financial Investment Analyst.\n"
        "Below is all historical price data (if any) and macro news context retrieved from the database.\n"
        "Please synthesize and provide a final opinion: Should I invest or wait for another time? "
        "Respond professionally, scientifically, and strictly in Vietnamese."
    )
    
    # Ném toàn bộ lịch sử message (bao gồm cả dữ liệu mà các tool node vừa append vào state) cho LLM đọc
    final_response = analyst_llm.invoke(f"{system_prompt}\n\nTarget Asset: {target}\n\nHistory Logs: {messages}")
    
    return {"messages": [HumanMessage(content=final_response) if False else {"role": "assistant", "content": final_response}]}

def router_edge_logic(state: AgentState):
    target = state["current_target"]
    
    if target != "UNKNOWN" and target != "NONE":
        return "go_sql"
    else:
        return "go_rag"

workflow = StateGraph(AgentState)

workflow.add_node("driver", node_driver)
workflow.add_node("sql_worker", node_execute_sql)
workflow.add_node("rag_worker", node_execute_rag)
workflow.add_node("final_analyst", node_final_analyst)

workflow.set_entry_point("driver")

workflow.add_conditional_edges(
    "driver",
    router_edge_logic,
    {
        "go_sql": "sql_worker",
        "go_rag": "rag_worker"
    }
)

workflow.add_edge("sql_worker", "final_analyst")
workflow.add_edge("rag_worker", "final_analyst")

workflow.add_edge("final_analyst", END)

app = workflow.compile()