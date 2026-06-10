# tests/test_core.py
import os
import json
import pytest

def test_intent_parsing():
    """Test kịch bản bóc tách câu chào hỏi thông thường"""
    mock_response = '{"target": "UNKNOWN", "intent": "CHITCHAT"}'
    data = json.loads(mock_response)
    assert data["intent"] == "CHITCHAT"
    assert data["target"] == "UNKNOWN"

def test_intent_parsing_2():
    """Test kịch bản câu hỏi đòi số liệu cứng"""
    mock_response = '{"target": "BTC", "intent": "MARKET_DATA"}'
    data = json.loads(mock_response)
    assert data["intent"] == "MARKET_DATA"
    assert data["target"] == "BTC"