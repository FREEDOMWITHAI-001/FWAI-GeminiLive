"""
Book Demo Tool
Books a demo/consultation slot
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry


@ToolRegistry.register
class BookDemoTool(BaseTool):
    name = "book_demo"
    description = "Book a demo or consultation session for the Gold Membership program"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the person booking"
                },
                "preferred_date": {
                    "type": "string",
                    "description": "Preferred date for demo (e.g., 'tomorrow', 'next Monday', 'January 25')"
                },
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred time slot (e.g., 'morning', '2 PM', 'evening')"
                },
                "interest_area": {
                    "type": "string",
                    "description": "What they're most interested in learning about"
                }
            },
            "required": ["preferred_date", "preferred_time"]
        }
    
    async def execute(self, caller_phone: str, preferred_date: str, preferred_time: str, 
                      name: str = "", interest_area: str = "", **kwargs) -> ToolResult:
        """Book a demo session"""
        try:
            # Store demo booking
            demos_file = Path(__file__).parent.parent.parent / "data" / "demo_bookings.json"
            demos_file.parent.mkdir(exist_ok=True)
            
            # Load existing bookings
            bookings = []
            if demos_file.exists():
                with open(demos_file, 'r') as f:
                    bookings = json.load(f)
            
            # Add new booking
            booking = {
                "id": len(bookings) + 1,
                "phone": caller_phone,
                "name": name,
                "preferred_date": preferred_date,
                "preferred_time": preferred_time,
                "interest_area": interest_area,
                "status": "confirmed",
                "created_at": datetime.now().isoformat()
            }
            bookings.append(booking)
            
            # Save bookings
            with open(demos_file, 'w') as f:
                json.dump(bookings, f, indent=2)
            
            logger.info(f"Demo booked for {caller_phone}: {preferred_date} at {preferred_time}")
            return ToolResult(
                success=True,
                message=f"Demo session booked for {preferred_date} at {preferred_time}",
                data=booking
            )
                    
        except Exception as e:
            logger.error(f"Book demo error: {e}")
            return ToolResult(
                success=False,
                message=f"Error booking demo: {str(e)}"
            )


# Convenience function
async def book_demo(caller_phone: str, preferred_date: str, preferred_time: str,
                    name: str = "", interest_area: str = "") -> ToolResult:
    tool = BookDemoTool()
    return await tool.execute(
        caller_phone=caller_phone,
        preferred_date=preferred_date,
        preferred_time=preferred_time,
        name=name,
        interest_area=interest_area
    )
