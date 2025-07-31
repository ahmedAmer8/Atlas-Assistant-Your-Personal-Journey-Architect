import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration settings"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Model Settings
    GEMINI_MODEL: str = "gemini-1.5-flash"
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    MAX_TOKENS: int = 2048
    
    # Database Settings
    VECTOR_DB_PATH: str = "data/travel_attractions_db"
    CONVERSATION_HISTORY_LIMIT: int = 20
    
    # Tool Settings
    WEATHER_FORECAST_DAYS: int = 7
    MAX_TOOL_RETRIES: int = 3
    TOOL_RETRY_DELAY: float = 1.0
    
    # Default Values
    DEFAULT_BUDGET_MARGIN: float = 0.15  # 15% buffer
    DEFAULT_ACTIVITY_DURATION: int = 2  # hours
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not self.OPENWEATHER_API_KEY:
            raise ValueError("OPENWEATHER_API_KEY environment variable is required")
        return True

# Create global config instance
config = Config()