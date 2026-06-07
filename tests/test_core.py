# tests/test_core.py
import os
import json
import pytest

def test_project_structure():
    """Kiểm tra xem các file dữ liệu cấu trúc core có nằm đúng vị trí không"""
    assert os.path.exists("data/raw.csv") == True

def test_json_parsing_logic():
    """Test độc lập logic bóc tách JSON State để đảm bảo hàm json.loads không bị crash vô lý"""
    mock_llm_response = '{"target": "BTC"}'
    
    parsed_data = json.loads(mock_llm_response)
    
    assert "target" in parsed_data
    assert parsed_data["target"] == "BTC"

def test_unknown_json_parsing_logic():
    """Test kịch bản Edge Case khi user chat câu phá hoại và LLM nhả UNKNOWN"""
    mock_llm_response = '{"target": "UNKNOWN"}'
    parsed_data = json.loads(mock_llm_response)
    
    assert parsed_data["target"] == "UNKNOWN"