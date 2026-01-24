"""
Tool Registry - Manages all available tools for the AI Agent
"""

from typing import Dict, List, Any, Optional
from loguru import logger

# Tool registry
_tools: Dict[str, Any] = {}


class ToolRegistry:
    """Registry for all available tools"""
    
    @classmethod
    def register(cls, tool_class):
        """Register a tool class"""
        instance = tool_class()
        _tools[instance.name] = instance
        logger.debug(f"Registered tool: {instance.name}")
        return tool_class
    
    @classmethod
    def get_tool(cls, name: str):
        """Get a tool by name"""
        return _tools.get(name)
    
    @classmethod
    def get_all_tools(cls) -> List[Any]:
        """Get all registered tools"""
        return list(_tools.values())
    
    @classmethod
    def get_tool_names(cls) -> List[str]:
        """Get all tool names"""
        return list(_tools.keys())


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get all tool definitions for Gemini"""
    tools = ToolRegistry.get_all_tools()
    return [tool.get_definition() for tool in tools]


async def execute_tool(tool_name: str, caller_phone: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name"""
    tool = ToolRegistry.get_tool(tool_name)
    
    if not tool:
        logger.error(f"Tool not found: {tool_name}")
        return {
            "success": False,
            "message": f"Tool '{tool_name}' not found"
        }
    
    try:
        logger.info(f"Executing tool: {tool_name} for {caller_phone}")
        result = await tool.execute(caller_phone=caller_phone, **kwargs)
        logger.info(f"Tool result: {result}")
        return {
            "success": result.success,
            "message": result.message,
            "data": result.data
        }
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {
            "success": False,
            "message": f"Error executing tool: {str(e)}"
        }
