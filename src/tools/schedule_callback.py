"""
Schedule Callback Tool
Stores callback requests for follow-up
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry


@ToolRegistry.register
class ScheduleCallbackTool(BaseTool):
    name = "schedule_callback"
    description = "Schedule a callback for the caller at a preferred time"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred callback time (e.g., 'tomorrow morning', '3 PM today', 'Monday 10 AM')"
                },
                "notes": {
                    "type": "string",
                    "description": "Any notes about what to discuss on callback"
                }
            },
            "required": ["preferred_time"]
        }
    
    async def execute(self, caller_phone: str, preferred_time: str, notes: str = "", **kwargs) -> ToolResult:
        """Schedule a callback"""
        try:
            # Store callback request
            callbacks_file = Path(__file__).parent.parent.parent / "data" / "callbacks.json"
            callbacks_file.parent.mkdir(exist_ok=True)
            
            # Load existing callbacks
            callbacks = []
            if callbacks_file.exists():
                with open(callbacks_file, 'r') as f:
                    callbacks = json.load(f)
            
            # Add new callback
            callback = {
                "id": len(callbacks) + 1,
                "phone": caller_phone,
                "preferred_time": preferred_time,
                "notes": notes,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            callbacks.append(callback)
            
            # Save callbacks
            with open(callbacks_file, 'w') as f:
                json.dump(callbacks, f, indent=2)
            
            logger.info(f"Callback scheduled for {caller_phone} at {preferred_time}")
            return ToolResult(
                success=True,
                message=f"Callback scheduled for {preferred_time}",
                data=callback
            )
                    
        except Exception as e:
            logger.error(f"Schedule callback error: {e}")
            return ToolResult(
                success=False,
                message=f"Error scheduling callback: {str(e)}"
            )


# Convenience function
async def schedule_callback(caller_phone: str, preferred_time: str, notes: str = "") -> ToolResult:
    tool = ScheduleCallbackTool()
    return await tool.execute(
        caller_phone=caller_phone,
        preferred_time=preferred_time,
        notes=notes
    )
