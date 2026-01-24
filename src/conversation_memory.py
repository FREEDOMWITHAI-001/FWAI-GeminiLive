"""
Conversation Memory for Plivo Calls
Stores conversation history per call_uuid
"""

from typing import Dict, List
from datetime import datetime, timedelta

# In-memory storage for conversations
conversations: Dict[str, List[Dict]] = {}

def add_message(call_uuid: str, role: str, content: str):
    """Add a message to conversation history"""
    if call_uuid not in conversations:
        conversations[call_uuid] = []
    
    conversations[call_uuid].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def get_history(call_uuid: str) -> str:
    """Get formatted conversation history for Gemini"""
    if call_uuid not in conversations:
        return ""
    
    history = []
    for msg in conversations[call_uuid]:
        if msg["role"] == "user":
            history.append(f"User: {msg.get('content')}")
        else:
            history.append(f"Vishnu: {msg.get('content')}")
    
    return "\n".join(history)

def clear_conversation(call_uuid: str):
    """Clear conversation when call ends"""
    if call_uuid in conversations:
        del conversations[call_uuid]

def get_turn_count(call_uuid: str) -> int:
    """Get number of turns in conversation"""
    if call_uuid not in conversations:
        return 0
    return len([m for m in conversations[call_uuid] if m["role"] == "user"])
