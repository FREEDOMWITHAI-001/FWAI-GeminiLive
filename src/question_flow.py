# Question Flow State Machine
# Manages conversation flow internally - passes ONE question at a time

import json
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Directory for saving flow data (persists across disconnects)
FLOW_DATA_DIR = Path(__file__).parent.parent / "flow_data"
FLOW_DATA_DIR.mkdir(exist_ok=True)


@dataclass
class QuestionFlow:
    """
    Tracks conversation state and provides next question.
    Each call session gets its own QuestionFlow instance.
    """
    call_uuid: str = ""
    customer_name: str = "there"
    agent_name: str = "Rahul"
    current_step: int = 0
    collected_data: Dict = field(default_factory=dict)

    # Questions list - simple and ordered
    QUESTIONS: List[Dict] = field(default_factory=lambda: [
        {
            "id": "opening",
            "prompt": "Hi {customer_name}, this is {agent_name} from Freedom with AI. You attended our AI Masterclass with Avinash Mada, right? How was it?"
        },
        {
            "id": "followup",
            "prompt": "That's interesting! What part resonated most with you?"
        },
        {
            "id": "expectations",
            "prompt": "Did you have any specific expectations from the masterclass?"
        },
        {
            "id": "source",
            "prompt": "How did you first hear about Freedom With AI?"
        },
        {
            "id": "knowledge",
            "prompt": "What do you know about what we do at FWAI?"
        },
        {
            "id": "ai_understanding",
            "prompt": "What's your understanding of AI and how it impacts businesses?"
        },
        {
            "id": "ai_rating",
            "prompt": "On a scale of 1 to 10, how would you rate your AI knowledge?"
        },
        {
            "id": "ai_usage",
            "prompt": "Are you using any AI tools like ChatGPT in your work?"
        },
        {
            "id": "ai_importance",
            "prompt": "Do you feel it's important to learn AI? How are you planning to?"
        },
        {
            "id": "goal_6months",
            "prompt": "If AI could help you achieve one result in 6 months, what would it be?"
        },
        {
            "id": "about_you",
            "prompt": "Tell me about yourself - what do you do currently?"
        },
        {
            "id": "career_goal",
            "prompt": "What's your long-term career goal?"
        },
        {
            "id": "roadmap",
            "prompt": "Do you follow any roadmap or mentor for career growth?"
        },
        {
            "id": "presentation",
            "prompt": "Based on what you shared, our AI Upskilling Program seems perfect. It has 12 modules, 300+ AI tools, hands-on projects. Self-paced with weekend live classes. Does this interest you?"
        },
        {
            "id": "closing",
            "prompt": "Great talking to you! I'll share all the details on WhatsApp. When's a good time to call back - morning or evening?"
        }
    ])

    # Objection responses
    OBJECTIONS: Dict[str, str] = field(default_factory=lambda: {
        "price": "It's 40K but masterclass attendees get a discount. EMI available too. Does that help?",
        "family": "Of course, discuss with family. I'll WhatsApp details. Call back morning or evening?",
        "exams": "No problem! It's self-paced, pause during exams. When do they end?",
        "busy": "No worries! When's a better time to call?",
        "youtube": "YouTube is great for basics. We add structure, projects, and certification. Want to know more?",
        "not_interested": "Thanks for your time, {customer_name}. Reach out when ready. Take care!"
    })

    def get_base_prompt(self) -> str:
        """Minimal base prompt - just voice and rules"""
        return f"""You are {self.agent_name}, AI Counselor at Freedom with AI, Hyderabad.

VOICE: Indian English. Use "Actually...", "right?" naturally. NO Hindi. Be warm.

RULES:
1. Say ONLY what is given in [SAY THIS]. Do not add extra questions.
2. After speaking, STOP and WAIT for user response.
3. Keep it SHORT - max 2 sentences.
4. If user says bye: "Great talking to you, take care!" then use end_call tool.
5. If asked if AI: "No no, I'm {self.agent_name} from Hyderabad office."

EMPATHY PHRASES: "I understand", "That makes sense", "Got it"
"""

    def get_current_question(self) -> Optional[str]:
        """Get the current question to ask"""
        if self.current_step >= len(self.QUESTIONS):
            return None

        q = self.QUESTIONS[self.current_step]
        prompt = q["prompt"].format(
            customer_name=self.customer_name,
            agent_name=self.agent_name
        )
        return prompt

    def get_instruction_prompt(self) -> str:
        """Get the full instruction for current step"""
        question = self.get_current_question()
        if not question:
            return "[SAY THIS]: Great talking to you! Take care!"

        step_num = self.current_step + 1
        total = len(self.QUESTIONS)

        return f"""[QUESTION {step_num}/{total}]
[SAY THIS]: {question}
[THEN WAIT FOR USER RESPONSE - DO NOT CONTINUE TALKING]"""

    def detect_objection(self, user_text: str) -> Optional[str]:
        """Detect if user raised an objection"""
        text = user_text.lower()

        if any(w in text for w in ["price", "cost", "expensive", "afford", "how much"]):
            return "price"
        if any(w in text for w in ["family", "father", "mother", "parent", "husband", "wife", "discuss"]):
            return "family"
        if any(w in text for w in ["exam", "test", "semester"]) and not "interest" in text:
            return "exams"
        if any(w in text for w in ["busy", "call later", "not now", "bad time"]):
            return "busy"
        if any(w in text for w in ["youtube", "free", "learn myself"]):
            return "youtube"
        if any(w in text for w in ["not interested", "no thanks", "don't want"]):
            return "not_interested"

        return None

    def get_objection_response(self, objection_type: str) -> str:
        """Get response for an objection"""
        response = self.OBJECTIONS.get(objection_type, "I understand. Let me help you with that.")
        return response.format(customer_name=self.customer_name)

    def advance(self, user_response: str = "") -> str:
        """
        Process user response and get next instruction.
        Returns the prompt to inject into AI.
        """
        # Check for objection first
        objection = self.detect_objection(user_response)
        if objection:
            logger.info(f"Detected objection: {objection}")
            if objection == "not_interested":
                # End call
                return f"[SAY THIS]: {self.get_objection_response(objection)}\n[THEN USE end_call TOOL]"
            else:
                # Handle objection, don't advance
                return f"[SAY THIS]: {self.get_objection_response(objection)}\n[WAIT FOR RESPONSE]"

        # Store response data
        if self.current_step < len(self.QUESTIONS):
            q_id = self.QUESTIONS[self.current_step]["id"]
            self.collected_data[q_id] = user_response

        # Move to next question
        self.current_step += 1

        # Save after each response (survives disconnects/crashes)
        if self.call_uuid:
            self.save_to_file(self.call_uuid)

        # Check if done
        if self.current_step >= len(self.QUESTIONS):
            return "[SAY THIS]: Great talking to you, {customer_name}! I'll send details on WhatsApp. Take care!\n[THEN USE end_call TOOL]".format(
                customer_name=self.customer_name
            )

        return self.get_instruction_prompt()

    def get_collected_data(self) -> Dict:
        """Get all collected data from the conversation"""
        return {
            "customer_name": self.customer_name,
            "agent_name": self.agent_name,
            "current_step": self.current_step,
            "total_steps": len(self.QUESTIONS),
            "completed": self.current_step >= len(self.QUESTIONS),
            "responses": self.collected_data
        }

    def save_to_file(self, call_uuid: str):
        """Save current state to file (survives crashes/disconnects)"""
        try:
            data = self.get_collected_data()
            data["call_uuid"] = call_uuid
            file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[{call_uuid[:8]}] Saved flow data to file")
        except Exception as e:
            logger.error(f"Error saving flow data: {e}")


# Store flows per call
_call_flows: Dict[str, QuestionFlow] = {}


def get_or_create_flow(call_uuid: str, customer_name: str = "there", agent_name: str = "Rahul") -> QuestionFlow:
    """Get existing flow or create new one for a call"""
    if call_uuid not in _call_flows:
        _call_flows[call_uuid] = QuestionFlow(
            call_uuid=call_uuid,
            customer_name=customer_name,
            agent_name=agent_name
        )
        logger.info(f"[{call_uuid[:8]}] Created new QuestionFlow for {customer_name}")
    return _call_flows[call_uuid]


def remove_flow(call_uuid: str) -> Optional[Dict]:
    """Remove flow from memory and return collected data"""
    if call_uuid in _call_flows:
        flow = _call_flows.pop(call_uuid)
        data = flow.get_collected_data()
        # Save final state to file
        flow.save_to_file(call_uuid)
        return data
    return None


def get_flow_data_from_file(call_uuid: str) -> Optional[Dict]:
    """Load flow data from file (for analytics/recovery)"""
    try:
        file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading flow data: {e}")
    return None
