from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Individual conversation message"""
    role: str  
    content: str
    timestamp: str
    metadata: Optional[Dict] = None

@dataclass
class UserPreferences:
    """User travel preferences and constraints"""
    interests: List[str]
    budget_range: Tuple[float, float]  # (min, max)
    preferred_categories: List[str]
    avoided_categories: List[str]
    dietary_restrictions: List[str]
    mobility_constraints: List[str]
    language: str
    travel_style: str  # budget, mid_range, luxury
    group_size: int
    last_updated: str

class ConversationMemory:
    """Manages conversation history and user preferences"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.messages: List[Message] = []
        self.user_preferences: Optional[UserPreferences] = None
        self.session_context: Dict = {}
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation history"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        if len(self.messages) > self.max_history:
            self._truncate_history()
        
        if role == "user":
            self._extract_preferences(content)
        
        logger.info(f"Added {role} message to conversation history")
    
    def _truncate_history(self) -> None:
        """Intelligently truncate conversation history"""
        if len(self.messages) <= self.max_history:
            return
        
        recent_messages = self.messages[-self.max_history:]
        
        important_messages = []
        for msg in self.messages[:-self.max_history]:
            if self._is_important_message(msg):
                important_messages.append(msg)
        
        if important_messages:
            available_slots = self.max_history - len(recent_messages)
            important_messages = important_messages[-available_slots:] if available_slots > 0 else []
        
        self.messages = important_messages + recent_messages
        logger.info(f"Truncated conversation history to {len(self.messages)} messages")
    
    def _is_important_message(self, message: Message) -> bool:
        """Determine if a message contains important context"""
        important_keywords = [
            "budget", "prefer", "don't like", "interested in", "avoid",
            "allergic", "dietary", "wheelchair", "mobility", "group",
            "traveling with", "style", "luxury", "budget", "backpack"
        ]
        
        content_lower = message.content.lower()
        return any(keyword in content_lower for keyword in important_keywords)
    
    def _extract_preferences(self, user_input: str) -> None:
        """Extract and update user preferences from conversation"""
        if not self.user_preferences:
            self.user_preferences = UserPreferences(
                interests=[],
                budget_range=(0, 1000),
                preferred_categories=[],
                avoided_categories=[],
                dietary_restrictions=[],
                mobility_constraints=[],
                language=self._detect_language(user_input),
                travel_style="mid_range",
                group_size=1,
                last_updated=datetime.now().isoformat()
            )
        
        user_input_lower = user_input.lower()
        
        interest_keywords = {
            "history": ["history", "historical", "ancient", "heritage"],
            "art": ["art", "gallery", "museum", "painting", "sculpture"],
            "food": ["food", "restaurant", "cuisine", "eating", "dining"],
            "nature": ["nature", "park", "garden", "outdoor", "hiking"],
            "culture": ["culture", "cultural", "traditional", "local"],
            "nightlife": ["nightlife", "bar", "club", "evening", "night"],
            "shopping": ["shopping", "market", "boutique", "souvenir"],
            "architecture": ["architecture", "building", "cathedral", "mosque"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if interest not in self.user_preferences.interests:
                    self.user_preferences.interests.append(interest)
        
        if "budget" in user_input_lower:
            budget_numbers = self._extract_numbers(user_input)
            if budget_numbers:
                if len(budget_numbers) == 1:
                    self.user_preferences.budget_range = (0, budget_numbers[0])
                elif len(budget_numbers) >= 2:
                    self.user_preferences.budget_range = (budget_numbers[0], budget_numbers[1])
        
        dietary_keywords = ["vegetarian", "vegan", "halal", "kosher", "gluten-free", "allergic"]
        for keyword in dietary_keywords:
            if keyword in user_input_lower and keyword not in self.user_preferences.dietary_restrictions:
                self.user_preferences.dietary_restrictions.append(keyword)
        
        if any(word in user_input_lower for word in ["luxury", "high-end", "premium"]):
            self.user_preferences.travel_style = "luxury"
        elif any(word in user_input_lower for word in ["budget", "cheap", "backpack", "hostel"]):
            self.user_preferences.travel_style = "budget"
        
        group_indicators = ["traveling with", "group of", "family of", "couple", "solo"]
        for indicator in group_indicators:
            if indicator in user_input_lower:
                numbers = self._extract_numbers(user_input)
                if numbers:
                    self.user_preferences.group_size = numbers[0]
                elif "couple" in user_input_lower:
                    self.user_preferences.group_size = 2
                elif "solo" in user_input_lower:
                    self.user_preferences.group_size = 1
        
        self.user_preferences.last_updated = datetime.now().isoformat()
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(text) * 0.1:  # If more than 10% Arabic characters
            return "ar"
        return "en"
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text"""
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(num) for num in numbers]
    
    def get_conversation_history(self, format_for_model: bool = True) -> List[Dict]:
        """Get conversation history formatted for the model"""
        if format_for_model:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages
            ]
        return [asdict(msg) for msg in self.messages]
    
    def get_user_preferences(self) -> Optional[Dict]:
        """Get current user preferences"""
        if self.user_preferences:
            return asdict(self.user_preferences)
        return None
    
    def update_session_context(self, key: str, value: any) -> None:
        """Update session-specific context"""
        self.session_context[key] = value
        logger.info(f"Updated session context: {key}")
    
    def get_session_context(self, key: str) -> any:
        """Get session context value"""
        return self.session_context.get(key)
    
    def clear_session(self) -> None:
        """Clear session data while preserving learned preferences"""
        self.messages.clear()
        self.session_context.clear()
        logger.info("Session cleared")
    
    def get_context_summary(self) -> str:
        """Generate a summary of current context for the model"""
        summary_parts = []
        
        if self.user_preferences:
            prefs = self.user_preferences
            
            if prefs.interests:
                summary_parts.append(f"User interests: {', '.join(prefs.interests)}")
            
            if prefs.budget_range != (0, 1000):
                summary_parts.append(f"Budget range: ${prefs.budget_range[0]}-${prefs.budget_range[1]}")
            
            if prefs.travel_style != "mid_range":
                summary_parts.append(f"Travel style: {prefs.travel_style}")
            
            if prefs.group_size != 1:
                summary_parts.append(f"Group size: {prefs.group_size}")
            
            if prefs.dietary_restrictions:
                summary_parts.append(f"Dietary restrictions: {', '.join(prefs.dietary_restrictions)}")
        
        if self.session_context:
            if 'current_destination' in self.session_context:
                summary_parts.append(f"Current destination: {self.session_context['current_destination']}")
        
        return "; ".join(summary_parts) if summary_parts else "No specific preferences recorded yet"
    
    def export_memory(self) -> Dict:
        """Export all memory data"""
        return {
            "messages": [asdict(msg) for msg in self.messages],
            "user_preferences": asdict(self.user_preferences) if self.user_preferences else None,
            "session_context": self.session_context
        }
    
    def import_memory(self, data: Dict) -> None:
        """Import memory data"""
        if "messages" in data:
            self.messages = [Message(**msg_data) for msg_data in data["messages"]]
        
        if "user_preferences" in data and data["user_preferences"]:
            self.user_preferences = UserPreferences(**data["user_preferences"])
        
        if "session_context" in data:
            self.session_context = data["session_context"]
        
        logger.info("Memory data imported successfully")