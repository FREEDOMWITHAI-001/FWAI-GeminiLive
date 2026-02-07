# Question Flow State Machine
# Manages conversation flow internally - passes ONE question at a time
# Loads configuration from client-specific JSON files

import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Directories
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
FLOW_DATA_DIR = Path(__file__).parent.parent / "flow_data"
FLOW_DATA_DIR.mkdir(exist_ok=True)


def load_client_config(client_name: str = "fwai") -> Dict[str, Any]:
    """Load client configuration from JSON file"""
    config_file = PROMPTS_DIR / f"{client_name}_config.json"

    if not config_file.exists():
        # Try default
        config_file = PROMPTS_DIR / "fwai_config.json"
        if not config_file.exists():
            logger.warning(f"No config found for {client_name}, using defaults")
            return get_default_config()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Loaded config from {config_file.name}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Default configuration if no file found"""
    return {
        "defaults": {
            "agent_name": "Assistant",
            "company_name": "Our Company",
            "location": "Office",
            "voice": "Puck"
        },
        "base_prompt": "You are {agent_name} from {company_name}. Be helpful and professional.",
        "questions": [
            {"id": "greeting", "prompt": "Hello {customer_name}, how can I help you today?"}
        ],
        "objections": {},
        "objection_keywords": {}
    }


@dataclass
class QuestionFlow:
    """
    Tracks conversation state and provides next question.
    Loads questions from client-specific config file.
    """
    call_uuid: str = ""
    client_name: str = "fwai"
    context: Dict = field(default_factory=dict)
    current_step: int = 0
    collected_data: Dict = field(default_factory=dict)

    # Loaded from config
    config: Dict = field(default_factory=dict)
    questions: List[Dict] = field(default_factory=list)
    objections: Dict[str, str] = field(default_factory=dict)
    objection_keywords: Dict[str, List[str]] = field(default_factory=dict)
    base_prompt: str = ""

    def __post_init__(self):
        """Load config after initialization"""
        self.config = load_client_config(self.client_name)

        # Merge defaults with provided context
        defaults = self.config.get("defaults", {})
        self.context = {**defaults, **self.context}

        # Load questions and objections
        self.questions = self.config.get("questions", [])
        self.objections = self.config.get("objections", {})
        self.objection_keywords = self.config.get("objection_keywords", {})
        self.base_prompt = self.config.get("base_prompt", "")

        logger.debug(f"Loaded {len(self.questions)} questions for {self.client_name}")

    def _render(self, template: str) -> str:
        """Replace {placeholders} with context values"""
        result = template
        for key, value in self.context.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def get_base_prompt(self) -> str:
        """Get the base prompt with placeholders replaced"""
        return self._render(self.base_prompt)

    def get_voice(self) -> str:
        """Get the voice setting from config"""
        return self.context.get("voice", "Puck")

    def get_current_question(self) -> Optional[str]:
        """Get the current question to ask"""
        if self.current_step >= len(self.questions):
            return None

        q = self.questions[self.current_step]
        return self._render(q["prompt"])

    def get_instruction_prompt(self) -> str:
        """Get the full instruction for current step"""
        question = self.get_current_question()
        if not question:
            return "[SAY THIS]: Great talking to you! Take care!"

        step_num = self.current_step + 1
        total = len(self.questions)

        return f"""The customer is waiting. Say exactly this to them: "{question}" Then stop talking and wait for their reply."""

    def detect_objection(self, user_text: str) -> Optional[str]:
        """Detect if user raised an objection"""
        text = user_text.lower()

        for objection_type, keywords in self.objection_keywords.items():
            # Special case: "exam" shouldn't match "interest"
            if objection_type == "exams" and "interest" in text:
                continue

            if any(kw in text for kw in keywords):
                return objection_type

        return None

    def get_objection_response(self, objection_type: str) -> str:
        """Get response for an objection"""
        response = self.objections.get(objection_type, "I understand. Let me help with that.")
        return self._render(response)

    def advance(self, user_response: str = "") -> str:
        """
        Process user response and get next instruction.
        Returns the prompt to inject into AI.
        """
        # Check for objection first
        objection = self.detect_objection(user_response)
        if objection:
            logger.debug(f"Objection: {objection}")
            if objection == "not_interested":
                return f"[SAY THIS]: {self.get_objection_response(objection)}\n[THEN USE end_call TOOL]"
            else:
                # Handle objection, don't advance
                return f"[SAY THIS]: {self.get_objection_response(objection)}\n[WAIT FOR RESPONSE]"

        # Store response data
        if self.current_step < len(self.questions):
            q_id = self.questions[self.current_step]["id"]
            self.collected_data[q_id] = user_response

        # Move to next question
        self.current_step += 1

        # Save after each response (survives disconnects)
        if self.call_uuid:
            self.save_to_file(self.call_uuid)

        # Check if done
        if self.current_step >= len(self.questions):
            closing = self._render("Great talking to you, {customer_name}! I'll send details on WhatsApp. Take care!")
            return f"[SAY THIS]: {closing}\n[THEN USE end_call TOOL]"

        return self.get_instruction_prompt()

    def get_collected_data(self) -> Dict:
        """Get all collected data from the conversation"""
        return {
            "call_uuid": self.call_uuid,
            "client_name": self.client_name,
            "customer_name": self.context.get("customer_name", ""),
            "agent_name": self.context.get("agent_name", ""),
            "current_step": self.current_step,
            "total_steps": len(self.questions),
            "completed": self.current_step >= len(self.questions),
            "responses": self.collected_data
        }

    def save_to_file(self, call_uuid: str):
        """Save current state to file"""
        try:
            data = self.get_collected_data()
            file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved flow data")
        except Exception as e:
            logger.error(f"Error saving flow data: {e}")


# Store flows per call
_call_flows: Dict[str, QuestionFlow] = {}


def get_or_create_flow(
    call_uuid: str,
    client_name: str = "fwai",
    context: Dict = None
) -> QuestionFlow:
    """Get existing flow or create new one for a call"""
    if call_uuid not in _call_flows:
        _call_flows[call_uuid] = QuestionFlow(
            call_uuid=call_uuid,
            client_name=client_name,
            context=context or {}
        )
    return _call_flows[call_uuid]


def remove_flow(call_uuid: str) -> Optional[Dict]:
    """Remove flow from memory and return collected data"""
    if call_uuid in _call_flows:
        flow = _call_flows.pop(call_uuid)
        data = flow.get_collected_data()
        flow.save_to_file(call_uuid)
        return data
    return None


def get_flow_data_from_file(call_uuid: str) -> Optional[Dict]:
    """Load flow data from file"""
    try:
        file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading flow data: {e}")
    return None
