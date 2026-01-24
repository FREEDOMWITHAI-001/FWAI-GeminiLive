"""
Base Tool Class for AI Agent Tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """Base class for all tools"""
    
    name: str = "base_tool"
    description: str = "Base tool description"
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(self, caller_phone: str, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def get_definition(self) -> Dict[str, Any]:
        """Get Gemini function declaration format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
