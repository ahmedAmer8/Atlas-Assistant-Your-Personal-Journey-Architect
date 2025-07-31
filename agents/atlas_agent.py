import logging
from typing import Dict, Optional, Any

from config import config
from llm.gemini_client import GeminiClient
from tools.weather_tool import WeatherTool
from tools.budget_calculator import BudgetCalculator
from memory.conversation_memory import ConversationMemory
from vector_db import PlaceFinder
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class AtlasAgent:
    """
    Main agent controller for Atlas Assistant
    Orchestrates all tools and manages conversation flow
    """
    
    def __init__(self):
        self.llm = GeminiClient(
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_MODEL,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS
        )
        
        self.weather_tool = WeatherTool(
            api_key=config.OPENWEATHER_API_KEY,
            retry_count=config.MAX_TOOL_RETRIES,
            retry_delay=config.TOOL_RETRY_DELAY
        )
        
        self.budget_calculator = BudgetCalculator(
            margin_percentage=config.DEFAULT_BUDGET_MARGIN
        )
        
        self.place_finder = PlaceFinder(config.VECTOR_DB_PATH)
        
        self.memory = ConversationMemory(
            max_history=config.CONVERSATION_HISTORY_LIMIT
        )
        
        self.system_prompt = """You are Atlas, a knowledgeable and friendly Travel Architect. Your role is to help users plan amazing travel experiences by combining practical information with inspiring suggestions.

                                PERSONALITY:
                                - Friendly and engaging, never using slang
                                - Balanced between inspiration and practicality
                                - Concise yet comprehensive
                                - Culturally aware and respectful
                                - Responds in the user's detected language (Arabic or English)

                                CAPABILITIES:
                                - Access real-time weather forecasts
                                - Search for attractions and places of interest
                                - Calculate detailed budgets and costs
                                - Remember user preferences throughout conversation
                                - Create day-by-day itineraries

                                RESPONSE FORMAT:
                                - Use clear headings for itineraries (Day 1 - Morning/Afternoon/Evening)
                                - Bullet-point activities with time estimates and costs
                                - End with budget summary and weather snapshot
                                - Stay factual about prices and weather

                                CONSTRAINTS:
                                - Never book anything - only suggest and recommend
                                - If data is missing, ask clarifying questions
                                - If tools fail, offer alternatives
                                - Be upfront about limitations
                                - Always consider user's stated preferences and budget

                                Remember: You're helping create memorable journeys, not just listing attractions.
        """

        logger.info("Atlas Agent initialized successfully")
    
    def process_message(self, user_input: str) -> str:
        """
        Process user message and generate response
        
        Args:
            user_input: User's input message
            
        Returns:
            Agent's response
        """
        try:
            self.memory.add_message("user", user_input)
            
            tool_needs = self._analyze_intent(user_input)
            
            tool_results = {}
            if tool_needs:
                tool_results = self._execute_tools(tool_needs, user_input)
            
            response = self._generate_response(tool_results)
            
            self.memory.add_message("assistant", response)
            
            return {
                "content": response,
                "metadata": tool_results
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
    
    def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user intent to determine which tools are needed"""
        intent = {
            "needs_weather": False,
            "needs_places": False,
            "needs_budget": False,
            "destination": None,
            "interests": [],
            "budget_limit": None
        }
        
        user_lower = user_input.lower()
        
        weather_keywords = ["weather", "temperature", "rain", "sunny", "climate", "forecast"]
        if any(keyword in user_lower for keyword in weather_keywords):
            intent["needs_weather"] = True
        
        place_keywords = ["attractions", "places", "visit", "see", "museum", "restaurant", 
                         "activities", "things to do", "sightseeing", "recommend"]
        if any(keyword in user_lower for keyword in place_keywords):
            intent["needs_places"] = True
        
        budget_keywords = ["budget", "cost", "price", "expensive", "cheap", "afford", "money"]
        if any(keyword in user_lower for keyword in budget_keywords):
            intent["needs_budget"] = True
        
        intent["destination"] = self._extract_destination(user_input)
        
        planning_keywords = ["itinerary", "plan", "trip", "travel", "schedule", "agenda"]
        if any(keyword in user_lower for keyword in planning_keywords):
            intent["needs_weather"] = True
            intent["needs_places"] = True
            intent["needs_budget"] = True
        
        return intent
    
    def _extract_destination(self, user_input: str) -> Optional[str]:
        """Extract destination from user input"""
        user_lower = user_input.lower()
        
        destination_patterns = [
            "to ", "in ", "visit ", "going to ", "traveling to ",
            "trip to ", "vacation in ", "holiday in "
        ]
        
        for pattern in destination_patterns:
            if pattern in user_lower:
                start_idx = user_lower.find(pattern) + len(pattern)
                remaining_text = user_input[start_idx:].strip()
                
                words = remaining_text.split()[:3]  # Take up to 3 words
                destination = " ".join(words).rstrip('.,!?')
                
                if len(destination) > 2:
                    return destination.title()
        
        current_dest = self.memory.get_session_context("current_destination")
        if current_dest:
            return current_dest
        
        return None
    
    def _execute_tools(self, tool_needs: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Execute required tools based on analyzed needs"""
        results = {}
        destination = tool_needs.get("destination")
        
        if destination:
            self.memory.update_session_context("current_destination", destination)
        
        if tool_needs["needs_weather"] and destination:
            logger.info(f"Fetching weather for {destination}")
            weather_data = self.weather_tool.get_weather_summary(
                destination, days=config.WEATHER_FORECAST_DAYS
            )
            results["weather"] = weather_data
        
        if tool_needs["needs_places"]:
            logger.info("Searching for places and attractions")
            
            search_query = self._build_place_search_query(user_input)
            
            places = self.place_finder.find_places(
                search_query, 
                limit=10,
                city_filter=destination if destination else None
            )
            results["places"] = places
            
        
        if tool_needs["needs_budget"] or results.get("places"):
            logger.info("Calculating budget estimates")
            
            self.budget_calculator.reset()
            
            user_prefs = self.memory.get_user_preferences()
            travel_style = "mid_range"
            if user_prefs:
                travel_style = user_prefs.get("travel_style", "mid_range")
            
            if results.get("places"):
                self.budget_calculator.add_attraction_costs(results["places"][:5])  # Top 5 attractions
            
            self.budget_calculator.add_meal_costs(travel_style, 3)  # 3 meals
            self.budget_calculator.add_transport_cost("local_transport", 1)
            
            budget_summary = self.budget_calculator.calculate_summary()
            results["budget"] = budget_summary.to_dict() if hasattr(budget_summary, 'to_dict') else budget_summary
        
        return results
    
    def _build_place_search_query(self, user_input: str) -> str:
        """Build search query for place finder"""
        user_prefs = self.memory.get_user_preferences()
        query_parts = []
        
        query_parts.append(user_input)
        
        if user_prefs and user_prefs.get("interests"):
            query_parts.extend(user_prefs["interests"])
        
        return " ".join(query_parts)
    
    def _generate_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate response using LLM with tool results"""
        conversation = self.memory.get_conversation_history()
        
        context_summary = self.memory.get_context_summary()
        enhanced_prompt = f"{self.system_prompt}\n\nUSER CONTEXT: {context_summary}"
        
        available_tools = list(tool_results.keys()) if tool_results else []
        
        response = self.llm.generate_with_tools(
            messages=conversation,
            system_prompt=enhanced_prompt,
            available_tools=available_tools,
            tool_results=tool_results
        )
        
        return response or "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            "total_messages": len(self.memory.messages),
            "user_preferences": self.memory.get_user_preferences(),
            "session_context": self.memory.session_context,
            "place_finder_stats": self.place_finder.db.get_statistics(),
            "current_budget": self.budget_calculator.to_dict()
        }
    
    def reset_session(self) -> None:
        """Reset current session while preserving learned preferences"""
        self.memory.clear_session()
        self.budget_calculator.reset()
        logger.info("Session reset completed")
    
    def set_creativity_level(self, level: str) -> None:
        """
        Adjust LLM creativity level
        
        Args:
            level: 'conservative', 'balanced', or 'creative'
        """
        creativity_settings = {
            "conservative": {"temperature": 0.3, "top_p": 0.8},
            "balanced": {"temperature": 0.7, "top_p": 0.9},
            "creative": {"temperature": 0.9, "top_p": 0.95}
        }
        
        if level in creativity_settings:
            settings = creativity_settings[level]
            self.llm.set_parameters(**settings)
            logger.info(f"Set creativity level to: {level}")
    
    def handle_error_gracefully(self, error: Exception, context: str) -> str:
        """Handle errors gracefully with user-friendly messages"""
        logger.error(f"Error in {context}: {error}")
        
        error_responses = {
            "weather": "I'm having trouble accessing weather information right now. I can still help you plan your activities based on typical weather patterns for the season.",
            "places": "I'm experiencing issues with the attractions database. I can provide general recommendations based on popular destinations.",
            "budget": "Budget calculations are temporarily unavailable, but I can provide rough cost estimates based on typical travel expenses."
        }
        
        return error_responses.get(context, "I encountered a technical issue, but I'm still here to help with your travel planning. Please try rephrasing your question.")