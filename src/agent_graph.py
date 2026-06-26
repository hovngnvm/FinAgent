from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM, ChatOllama
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis
from dotenv import load_dotenv
load_dotenv()

import json, re, os

# Đảm bảo các tool đã được cập nhật driver PostgreSQL từ Giai đoạn 1
from src.tools import tool_run_sqlite_query, tool_semantic_rag_search, tool_generate_market_chart

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    security_status: str
    current_target: str
    chat: bool
        
    sql_data_output: list[dict[str, any]]
    rag_text_output: str
    chart_status_msg: str
    
    error_log: str
    retry_count: int
    
    next_worker: str

# Khởi tạo mô hình ngôn ngữ lớn (Đồng bộ cấu hình vLLM/Ollama local)
llm = OllamaLLM(model="qwen2.5-coder:3b-instruct-q5_K_S", temperature=0, format="json")

def node_security_shield(state: AgentState):
    """Guardrails: Bảo vệ hệ thống khỏi Prompt Injection và độc hại"""
    last_user_message = state["messages"][-1].content
    
    SQL_INJECTION_REGEX = re.compile(
        r"(\b(DROP|DELETE|ALTER|TRUNCATE|UPDATE)\b\s+\b(TABLE|FROM|DATABASE|KEY)\b)|(--)|(\/\*|\*\/)",
        re.IGNORECASE
    )
    
    if SQL_INJECTION_REGEX.search(last_user_message):
        return {"security_status": "MALICIOUS", "next_worker": "FINISH"}
    
    llm_json = ChatOllama(model="llama-guard3:1b-q5_K_S", temperature=0.0).with_structured_output({
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["SAFE", "MALICIOUS"]}},
        "required": ["status"]
    })
    
    try:
        response = llm_json.invoke([{"role": "user", "content": last_user_message}])
        status = response.get("status", "SAFE")
    except Exception as e:
        status = "MALICIOUS"
        
    if status == "MALICIOUS":
        return {"security_status": "MALICIOUS", "next_worker": "FINISH"}
    else:
        return {"security_status": "SAFE", "next_worker": "CONTINUE"}

def node_driver(state: AgentState):
    """Supervisor Agent: Phân tích Intent ban đầu và điều phối Multi-Agent"""
    user_msg = state["messages"][0].content
    
    system_instruction = (
        "You are the Strategic Supervisor of the FinAgent system. Analyze the user's query.\n"
        "Determine the asset symbol (target ticker, e.g., 'BTC', 'ETH', 'HPG', or 'UNKNOWN').\n"
        "Categorize the query into one of two modes:\n"
        "1. 'CHITCHAT': If the query is a greeting, casual talk, or not related to financial investment analysis. Set chat to true.\n"
        "2. 'INVESTMENT': If the query requests stock/crypto analysis, price history, technical indicators, or investment opinions. Set chat to false.\n"
        "MANDATORY JSON OUTPUT FORMAT: {\"target\": \"SYMBOL\", \"mode\": \"INVESTMENT\" or \"CHITCHAT\"}"
    )
    
    response = llm.invoke(f"{system_instruction}\n\nUser Question: {user_msg}")
    
    try:
        parsed = json.loads(response)
        mode = parsed.get("mode", "CHITCHAT")
        return {
            "current_target": parsed.get("target", "UNKNOWN"),
            "chat": True if mode == "CHITCHAT" else False,
            "next_worker": "FINAL_ANALYST" if mode == "CHITCHAT" else "PARALLEL_EXECUTE"
        }
    except:
        return {"current_target": "UNKNOWN", "chat": True, "next_worker": "FINAL_ANALYST"}

def node_sql_worker(state: AgentState):
    """SQL Agent chuyên trách: Lấy dữ liệu định lượng từ PostgreSQL Gold Layer"""
    user_msg = state["messages"][0].content
    ticker = state["current_target"]
    retries = state.get("retry_count", 0)
    error = state.get("error_log", "")
    
    schema_info = "Table: prices | Columns: timestamp (TIMESTAMP), symbol (VARCHAR), price (DOUBLE PRECISION), volume (DOUBLE PRECISION), SMA_5, SMA_20, RSI, MACD, MACD_Signal"

    system_prompt = (
        "Based on the following PostgreSQL schema, please generate a high-performance SQL query:\n"
        f"Schema: {schema_info}\n"
        "You MUST return a JSON object with the following structure: {'sql': 'YOUR_SQL_QUERY'}\n"
        "Ensure your query directly filters the target asset's symbol, limits rows appropriately (max 30), and is valid syntax.\n"
        "Only return the JSON object, without any markdown code block wrap, text or explanation.\n"
    )

    if error:
        system_prompt += (
            f"\n\n[CRITICAL WARNING - FIX REQUIRED]:\n"
            f"Your previous SQL query FAILED with the following system error: '{error}'.\n"
            f"Please analyze the error, fix your syntax, and provide a corrected, valid SQLite query."
        )
        
    response = llm.invoke(f"{system_prompt}\n\nUser Question: {user_msg}")
    
    try:
        sql_query = json.loads(response).get("sql", f"SELECT * FROM prices WHERE symbol='{ticker}' ORDER BY timestamp DESC LIMIT 5;")
    except:
        sql_query = f"SELECT * FROM prices WHERE symbol='{ticker}' ORDER BY timestamp DESC LIMIT 5;"
        
    # Gọi hàm thực thi đã nâng cấp lên PostgreSQL ở giai đoạn 1
    clean_data, error_msg = tool_run_sqlite_query(sql_query)
    
    if error_msg:
        if retries + 1 < 2: # Self-correction Loop tối đa 2 lần
            return {"error_log": error_msg, "retry_count": retries + 1, "next_worker": "SQL_RETRY"}
        else:
            return {"error_log": f"PostgreSQL processing failure: {error_msg}", "sql_data_output": [], "next_worker": "JOIN_BARRIER"}
    
    return {"sql_data_output": clean_data, "error_log": "", "retry_count": 0, "next_worker": "JOIN_BARRIER"}

def node_rag_worker(state: AgentState):
    """RAG Agent chuyên trách: Khai phá văn bản vĩ mô từ Qdrant Vector DB"""
    user_msg = state["messages"][0].content
    
    # Kỹ thuật nâng cao: Giữ nguyên cơ chế kiểm tra và viết lại truy vấn (Query Rewriting) để tối ưu hóa vector search
    check_prompt = """
        Determine if the following user query contains excessive conversational filler or "noise" that obscures the core financial intent.
        Return a JSON object with a single boolean key 'need_rewrite' (true if noisy, false if concise and clear).
        
        Rules:
        - Focus on the user's financial intent. If the core intent is clear and direct, set 'need_rewrite' to false.
        - If the query contains significant conversational filler obscuring the financial intent, set 'need_rewrite' to true.
        - Return only the JSON object, without any additional text or explanation.
        
        JSON Output Format:
        {\"need_rewrite\": true/false}
    """
    try:
        check_res = llm.invoke(f"{check_prompt}\n\nUser: {user_msg}")
        need_rewrite = json.loads(check_res).get("need_rewrite", False)
    except:
        need_rewrite = False
        
    if need_rewrite:
        system_prompt = (
            "You are a Senior Context Engineering Expert. "
            "Your task is to read the user's financial query, analyze quickly the raw data from SQL "
            "and rewrite it into a HYPOTHETICAL DOCUMENT (HyDE) or an expanded keyword sequence (Query Expansion) containing specialized financial terms and removing the stop words and filler words. "
            "This enhanced text will be used to perform vector search, maximizing retrieval accuracy. "
            "Completely eliminate any redundant greetings or exclamations from the user. "
            "The output must be concise, professional, and focused on the economic essence."
        )
        query_to_search = llm.invoke(f"{system_prompt}\n\nUser Message: {user_msg}")
    else:
        query_to_search = user_msg
    
    rag_context = tool_semantic_rag_search(query_to_search)
    return {"rag_text_output": rag_context, "next_worker": "JOIN_BARRIER"}

def node_join_barrier(state: AgentState):
    """Barrier Node (Join): Đồng bộ hóa luồng dữ liệu song song trước khi phân tách nhánh đồ thị tiếp theo"""
    user_msg = state["messages"][0].content.lower()
    
    # Kiểm tra xem user có tường minh muốn vẽ/xem biểu đồ không
    chart_keywords = ["biểu đồ", "đồ thị", "vẽ", "chart", "graph", "visualize", "draw"]
    needs_chart = any(kw in user_msg for kw in chart_keywords)
    
    if needs_chart:
        return {"next_worker": "Chart_Agent"}
    else:
        return {"next_worker": "FINAL_ANALYST"}

def node_chart_worker(state: AgentState):
    """Chart Agent chuyên trách: Kết xuất đồ thị từ tập số liệu thu thập được"""
    ticker = state["current_target"]
    chart_result = tool_generate_market_chart(ticker)
    
    # SỬA LỖI LOGIC: Cập nhật biến trạng thái rõ ràng thay vì chỉ trả về ToolMessage
    return {
        "chart_status_msg": f"Success. {chart_result}", 
        "next_worker": "FINAL_ANALYST",
        "messages": [ToolMessage(content=chart_result, name="chart_generator", tool_call_id="chart_generator_call")]
    }

def node_final_analyst(state: AgentState):
    """Analyst Agent: Tổng hợp toàn bộ dữ liệu (SQL + RAG + Chart) đưa ra quyết định đầu tư cuối cùng"""
    llm_analyst = OllamaLLM(model="qwen3.5:4b-q4_K_M", temperature=0.3)
    
    conversation_history = state["messages"]
    user_msg = conversation_history[-1].content
        
    if state["chat"]:
        system_prompt = "You are a friendly financial assistant named FinAgent. Respond politely and naturally in Vietnamese."
        prompt_payload = f"=== USER QUESTION ===\n{user_msg}"
    else:
        target = state["current_target"]
        sql_numbers = state.get("sql_data_output", [])
        rag_news = state.get("rag_text_output", "No related macro news found.")
        chart_info = state.get("chart_status_msg", "No chart generated.")

        prompt_payload = (
            "TARGET ASSET PROFILE:\n"
            f"Target: {target}\n"
            "=== QUANTITATIVE HISTORICAL DATA (PostgreSQL) ===\n"
            f"{json.dumps(sql_numbers, indent=2) if sql_numbers else 'No time-series data available.'}\n\n"
            "=== QUALITATIVE NEWS CONTEXT (Qdrant RAG) ===\n"
            f"{rag_news}\n\n"
            "=== VISUALIZATION STATUS ===\n"
            f"{chart_info}\n\n"
            "=== CHAT HISTORY ===\n"
            f"{conversation_history}"
        )

        system_prompt = (
            "You are a Senior Financial Investment Analyst Expert.\n"
            "Synthesize the historical quantitative data, technical indicators, and qualitative news context provided.\n"
            "Provide a highly professional, definitive opinion: Should the user invest now, liquidate, or hold? \n"
            "Respond structurally, scientifically, and strictly in Vietnamese."
        )
    
    final_response = llm_analyst.invoke(f"{system_prompt}\n\n{prompt_payload}")
    return {"messages": [{"role": "assistant", "content": final_response}], "next_worker": "PURGE"}

def node_purge_state(state: AgentState):
    """State Purger: Làm sạch bộ nhớ đệm trạng thái giữa các phiên hội thoại để tối ưu VRAM"""
    return {
        "security_status": "",
        "chat": False,
        "sql_data_output": [],        
        "rag_text_output": "",          
        "chart_status_msg": "",        
        "error_log": "",                
        "retry_count": 0,
        "next_worker": "FINISH"
    }

# =====================================================================
# THIẾT LẬP ĐỒ THỊ LANGGRAPH ĐA TÁC TỬ SONG SONG (PARALLEL MULTI-AGENT WORKFLOW)
# =====================================================================
workflow = StateGraph(AgentState)

# Khai báo tất cả các Agent Node
workflow.add_node("security_shield", node_security_shield)
workflow.add_node("driver", node_driver)
workflow.add_node("sql_worker", node_sql_worker)
workflow.add_node("rag_worker", node_rag_worker)
workflow.add_node("join_barrier", node_join_barrier)
workflow.add_node("chart_worker", node_chart_worker)
workflow.add_node("final_analyst", node_final_analyst)
workflow.add_node("state_purger", node_purge_state)

workflow.set_entry_point("security_shield")

# Router 1: Gateway Security Check
workflow.add_conditional_edges(
    "security_shield",
    lambda state: "blocked" if state["next_worker"] == "FINISH" else "pass",
    {"blocked": "final_analyst", "pass": "driver"}
)

# Router 2: Supervisor Forking Router (Nhánh rẽ thông minh)
def supervisor_fork_router(state: AgentState):
    next_step = state["next_worker"]
    if next_step == "PARALLEL_EXECUTE":
        # Trả về mảng danh sách các node để kích hoạt cơ chế Parallel Fan-out song song thực sự
        return ["call_sql", "call_rag"] 
    else:
        return ["call_analyst_direct"]

workflow.add_conditional_edges(
    "driver",
    supervisor_fork_router,
    {
        "call_sql": "sql_worker",
        "call_rag": "rag_worker",
        "call_analyst_direct": "final_analyst"
    }
)

# Luồng tự sửa lỗi (Self-Correction Loop) riêng của SQL Worker
workflow.add_conditional_edges(
    "sql_worker",
    lambda state: "self_correction" if state["next_worker"] == "SQL_RETRY" else "to_barrier",
    {
        "self_correction": "sql_worker",
        "to_barrier": "join_barrier"
    }
)

# Nhánh RAG chạy xong tự động đi về điểm tụ (Barrier Node)
workflow.add_edge("rag_worker", "join_barrier")

# Router 3: Join Barrier Node (Fan-in Check)
workflow.add_conditional_edges(
    "join_barrier",
    lambda state: "call_chart" if state["next_worker"] == "Chart_Agent" else "call_analyst",
    {
        "call_chart": "chart_worker",
        "call_analyst": "final_analyst"
    }
)

# Điều hướng sau khi vẽ biểu đồ xong xuôi
workflow.add_edge("chart_worker", "final_analyst")

# Kết thúc vòng đời xử lý
workflow.add_edge("final_analyst", "state_purger")
workflow.add_edge("state_purger", END)

# Tích hợp cơ chế In-Memory Checkpointer phục vụ lưu trữ quản lý State Thread
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# 1. Khởi tạo một client Redis đồng bộ bền bỉ
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT)

# 2. Đóng gói client vào lớp Saver của LangGraph
redis_checkpointer = RedisSaver(redis_client)

# 3. Compile đồ thị sử dụng bộ nhớ ngoài Redis thay vì RAM nội bộ của Python
app = workflow.compile(checkpointer=redis_checkpointer)