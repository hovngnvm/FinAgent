from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM, ChatOllama
from langgraph.checkpoint.memory import MemorySaver

import json, re

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

llm = OllamaLLM(model="qwen2.5-coder:3b-instruct-q5_K_S", temperature=0, format="json") #json mode

def node_security_shield(state: AgentState):
    last_user_message = state["messages"][-1].content
    
    SQL_INJECTION_REGEX = re.compile(r"(\b(DROP|DELETE|ALTER|TRUNCATE|UPDATE)\b\s+\b(TABLE|FROM|DATABASE|KEY)\b)|(--)|(\/\*|\*\/)",
                                     re.IGNORECASE)
    
    if SQL_INJECTION_REGEX.search(last_user_message):
        return {"security_status": "MALICIOUS", "next_worker": "FINISH"}
    
    llm_json = ChatOllama(model="llama-guard3:1b-q5_K_S", temperature=0.0).with_structured_output({
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["SAFE", "MALICIOUS"]}},
        "required": ["status"]
    }) ####################
    
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
    user_msg = state["messages"][0].content
    sql_done = "YES" if state.get("sql_data_output") else "NO"
    rag_done = "YES" if state.get("rag_text_output") else "NO"
    chart_done = "YES" if state.get("chart_status_msg") else "NO"
    
    system_instruction = (
        "You are the Orchestration Manager of the FinAgent system. Read the user's query and the current progress status to decide which agent to call next.\n"
        "- 'SQL_Agent': Call when the user requests to view data, prices, or generate charts, and the system has not yet retrieved the data.\n"
        "- 'RAG_Agent': Call when the user asks about macroeconomics, trends, or after the SQL agent has completed data retrieval to find supporting information.\n"
        "- 'Chart_Agent': Call when the user requests to view charts/graphs AND the SQL agent has completed data retrieval.\n"
        "- 'FINISH': Call when sufficient data has been collected\n"
        "You have two modes: \"INVESTMENT\" and \"CHITCHAT\".\n"
        "- If the user asks about investment strategies, asset analysis, stock prices, financial data, or related topics, set \"chat\" to \"false\".\n"
        "- If the user asks about general topics, greetings, casual conversation, or non-investment questions, set \"chat\" to \"true\".\n"
        "MANDATORY JSON OUTPUT: {\"target\": \"SYMBOL\", \"assign_to\": \"AGENT_NAME\", \"chat\": true/false}"
    )
    
    status_context = f"Question: {user_msg}\nProgress:\n- SQL Done: {sql_done}\n- RAG Done: {rag_done}\n- Chart Done: {chart_done}"
    response = llm.invoke(f"{system_instruction}\n\nContext:\n{status_context}")
    
    try:
        parsed = json.loads(response)
        return {
            "current_target": parsed.get("target", "UNKNOWN"),
            "next_worker": parsed.get("assign_to", "FINISH")
        }
    except:
        return {"current_target": "UNKNOWN", "next_worker": "FINISH"}

def node_sql_worker(state: AgentState):
    user_msg = state["messages"][0].content
    ticker = state["current_target"]
    retries = state.get("retry_count", 0)
    error = state.get("error_log", "")
    
    schema_info = "Table: prices | Columns: timestamp (DATETIME), symbol (VARCHAR), price (REAL), volume (REAL), SMA_5(REAL), SMA_20(REAL), RSI(REAL), MACD(REAL), MACD_Signal(REAL)"

    system_prompt = (
        "Based on the following schema, please generate an accurate SQLite query:\n"
        f"Schema: {schema_info}\n"
        "You MUST return a JSON object with the following structure: {'sql': 'YOUR_SQL_QUERY'}\n"
        "Ensure your query is valid, efficient, and directly answers the user's request.\n"
        "Only return the JSON object, without any additional text or explanation.\n"
    )

    if error:
        system_prompt += (
            f"\n\n[CRITICAL WARNING - FIX REQUIRED]:\n"
            f"Your previous SQL query FAILED with the following system error: '{error}'.\n"
            f"Please analyze the error, fix your syntax, and provide a corrected, valid SQLite query."
        )
        
    response = llm.invoke(f"{system_prompt}\n\nUser Question: {user_msg}")
    
    sql_query = json.loads(response).get("sql", f"SELECT * FROM prices WHERE symbol='{ticker}' LIMIT 5;")
    
    clean_data, error_msg = tool_run_sqlite_query(sql_query)
    
    if error_msg != "":
        if retries + 1 < 2:
            return {"error_log": error_msg, "retry_count": retries + 1, "next_worker": "SQL_Agent"}
        else:
            return {"error_log": f"SQL system crashed: {error_msg}", "sql_data_output": [], "next_worker": "SUPERVISOR"}
    
    return {"sql_data_output": clean_data, "error_log": "", "retry_count": 0, "next_worker": "SUPERVISOR"}


def node_chart_worker(state: AgentState):
    ticker = state["current_target"]
    chart = tool_generate_market_chart(ticker)
    
    return {"messages": [ToolMessage(content=chart, name="chart_generator", tool_call_id="chart_generator_call")]}

def node_rag_worker(state: AgentState):
    user_msg = state["messages"][0].content
    sql_context = state.get("sql_data_output", [])
    
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
        payload = f"User: {user_msg}\nSQL Data: {json.dumps(sql_context[:3])}"
        query_to_search = llm.invoke(f"{system_prompt}\n\nPayload:\n{payload}")
    else:
        query_to_search = user_msg
    
    rag_context = tool_semantic_rag_search(query_to_search)
    return {"rag_text_output": rag_context, "next_worker": "SUPERVISOR"}

def node_final_analyst(state: AgentState):
    llm_analyst = OllamaLLM(model="qwen3.5:4b-q4_K_M", temperature=0.3)
    
    conversation_history = state["messages"]
    user_msg = conversation_history[-1].content
        
    if state["chat"]:
        system_prompt = (
            "You are a friendly finacial assistant aka FinAgent. respond politely and naturally in Vietnamese!"
        )
        prompt_payload = ("=== USER QUESTION ===\n" 
                         f"{user_msg}"
        )
    else:
        target = state["current_target"]
        sql_numbers = state.get("sql_data_output", [])
        rag_news = state.get("rag_text_output", "No related news found.")

        prompt_payload = (
            "TARGET ASSET PROFILE: \n"
            f"Target: {target}\n"
            "=== HISTORICAL DATA ===\n"
            f"{json.dumps(sql_numbers, indent=2) if sql_numbers else 'No time-series data available.'}\n\n"
            "=== NEWS CONTEXT ===\n"
            f"{rag_news}\n\n"
            "=== CHAT HISTORY ===\n"
            f"{conversation_history}"
    )

        system_prompt = (
        "You are a Senior Financial Investment Analyst.\n"
        "Below is all historical price data (if any) and macro news context retrieved from the database.\n"
        "Please synthesize and provide a final opinion: Should I invest or wait for another time? "
        "Respond professionally, scientifically, and strictly in Vietnamese."
    )
    
    final_response = llm_analyst.invoke(f"{system_prompt}\n\n{prompt_payload}")
    return {"messages": [{"role": "assistant", "content": final_response}], "next_worker": "PURGE"}

def node_purge_state(state: AgentState):
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

def supervisor_conditional_edge(state: AgentState):
    next_step = state["next_worker"]
    if next_step == "SQL_Agent": return "call_sql"
    elif next_step == "RAG_Agent": return "call_rag"
    elif next_step == "Chart_Agent": return "call_chart"
    else: return "call_analyst"

workflow = StateGraph(AgentState)

workflow.add_node("security_shield", node_security_shield)
workflow.add_node("driver", node_driver)
workflow.add_node("sql_worker", node_sql_worker)
workflow.add_node("chart_worker", node_chart_worker)
workflow.add_node("rag_worker", node_rag_worker)
workflow.add_node("final_analyst", node_final_analyst)
workflow.add_node("state_purger", node_purge_state)

workflow.set_entry_point("security_shield")


# Gateway Security Router
workflow.add_conditional_edges(
    "security_shield",
    lambda state: "blocked" if state["next_worker"] == "FINISH" else "pass",
    {"blocked": "final_analyst", "pass": "driver"}
)

workflow.add_conditional_edges(
    "driver",
    lambda state: "blocked" if state["next_worker"] == "FINISH" else "pass",
    {"blocked": "final_analyst", "pass": "supervisor"}
)

# Main Supervisor Router
workflow.add_conditional_edges(
    "driver",
    supervisor_conditional_edge,
    {
        "call_sql": "sql_worker",
        "call_rag": "rag_worker",
        "call_chart": "chart_worker",
        "call_analyst": "final_analyst"
    }
)

workflow.add_conditional_edges(
    "sql_worker",
    lambda state: "self_correction" if state["next_worker"] == "SQL_Agent" else "back_to_supervisor",
    {
        "self_correction": "sql_worker",
        "back_to_supervisor": "driver"
    }
)

workflow.add_edge("rag_worker", "driver")
workflow.add_edge("chart_worker", "driver")
workflow.add_edge("final_analyst", "state_purger")

workflow.add_edge("state_purger", END)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)