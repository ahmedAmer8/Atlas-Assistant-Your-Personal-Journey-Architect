import google.generativeai as genai
import time
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini API with retry logic and error handling"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro", 
                 temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    def generate_response(self, messages: List[Dict], system_prompt: str = "", 
                         retry_count: int = 3) -> Optional[str]:
        """
        Generate response from Gemini API
        
        Args:
            messages: List of conversation messages
            system_prompt: System instructions
            retry_count: Number of retry attempts
            
        Returns:
            Generated response or None if failed
        """
        conversation_text = self._format_conversation(messages, system_prompt)
        
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(
                    conversation_text,
                    safety_settings=self.safety_settings
                )
                
                if response.candidates[0].finish_reason.name == "SAFETY":
                    logger.warning("Response blocked by safety filters")
                    return "I apologize, but I cannot provide a response to that request. Please try rephrasing your question."
                
                return response.text
                
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("All Gemini API retry attempts failed")
                    return None
    
    def _format_conversation(self, messages: List[Dict], system_prompt: str) -> str:
        """Format conversation for Gemini API"""
        formatted_parts = []
        
        if system_prompt:
            formatted_parts.append(f"SYSTEM: {system_prompt}\n")
        
        for message in messages:
            role = "HUMAN" if message["role"] == "user" else "ASSISTANT"
            formatted_parts.append(f"{role}: {message['content']}\n")
        
        return "\n".join(formatted_parts)
    
    def generate_with_tools(self, messages: List[Dict], system_prompt: str,
                           available_tools: List[str], tool_results: Dict = None) -> str:
        """Generate response with tool integration context"""
        
        tool_context = self._create_tool_context(available_tools, tool_results)
        enhanced_system_prompt = f"{system_prompt}\n\n{tool_context}"
        
        return self.generate_response(messages, enhanced_system_prompt)
    
    def _create_tool_context(self, available_tools: List[str], tool_results: Dict = None) -> str:
        """Create context about available tools and their results"""
        context_parts = []
        
        if available_tools:
            context_parts.append("AVAILABLE TOOLS:")
            for tool in available_tools:
                context_parts.append(f"- {tool}")
        
        if tool_results:
            context_parts.append("\nTOOL RESULTS:")
            for tool_name, result in tool_results.items():
                if isinstance(result, dict) and 'error' in result:
                    context_parts.append(f"- {tool_name}: ERROR - {result['error']}")
                else:
                    # Truncate large results
                    result_str = str(result)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    context_parts.append(f"- {tool_name}: {result_str}")
        
        return "\n".join(context_parts)
    
    def set_parameters(self, temperature: float = None, top_p: float = None, 
                      max_tokens: int = None) -> None:
        """Update generation parameters"""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens
        
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
        )
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Updated generation parameters: temp={self.temperature}, top_p={self.top_p}")