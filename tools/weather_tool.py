import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather data structure"""
    date: str
    temperature_high: float
    temperature_low: float
    description: str
    humidity: int
    wind_speed: float
    precipitation_chance: int
    
class WeatherTool:
    """Weather API integration tool"""
    
    def __init__(self, api_key: str, retry_count: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"  # Remove /forecast
        self.retry_count = retry_count
        self.retry_delay = retry_delay
    
    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.retry_count):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Weather API attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error("All weather API retry attempts failed")
                    return None
    
    def get_coordinates(self, city: str, country: str = "") -> Optional[tuple]:
        """Get city coordinates using geocoding API"""
        location = f"{city},{country}" if country else city
        params = {
            'q': location,
            'appid': self.api_key,
            'limit': 1
        }
        
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct"
        data = self._make_request(geocoding_url, params)
        if data and len(data) > 0:
            return data[0]['lat'], data[0]['lon']
        return None
    
    def get_current_weather(self, city: str, country: str = "") -> Optional[Dict]:
        """Get current weather for a city"""
        location = f"{city},{country}" if country else city
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        data = self._make_request(f"{self.base_url}/weather", params)
        if not data:
            return None
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'description': data['weather'][0]['description'].title(),
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'visibility': data.get('visibility', 0) / 1000,  # Convert to km
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    
    def get_forecast(self, city: str, country: str = "", days: int = 7) -> List[WeatherData]:
        """Get weather forecast for specified days"""
        coords = self.get_coordinates(city, country)
        if not coords:
            return []
        
        lat, lon = coords
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        data = self._make_request(f"{self.base_url}/forecast", params)
        if not data or 'list' not in data:
            return []
        
        daily_forecasts = {}
        for item in data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        forecast_list = []
        for date_str in sorted(daily_forecasts.keys())[:days]:
            day_data = daily_forecasts[date_str]
            
            temps = [item['main']['temp'] for item in day_data]
            descriptions = [item['weather'][0]['description'] for item in day_data]
            humidity_vals = [item['main']['humidity'] for item in day_data]
            wind_speeds = [item['wind']['speed'] for item in day_data]
            precip_chances = [item.get('pop', 0) * 100 for item in day_data]
            
            weather_data = WeatherData(
                date=date_str,
                temperature_high=max(temps),
                temperature_low=min(temps),
                description=max(set(descriptions), key=descriptions.count).title(),
                humidity=int(sum(humidity_vals) / len(humidity_vals)),
                wind_speed=sum(wind_speeds) / len(wind_speeds),
                precipitation_chance=int(max(precip_chances))
            )
            forecast_list.append(weather_data)
        
        return forecast_list
    
    def get_weather_summary(self, city: str, country: str = "", days: int = 7) -> Dict:
        """Get comprehensive weather summary"""
        current = self.get_current_weather(city, country)
        forecast = self.get_forecast(city, country, days)
        
        if not current or not forecast:
            return {
                'error': f'Unable to fetch weather data for {city}',
                'suggestion': 'Please check the city name and try again'
            }
        
        return {
            'current': current,
            'forecast': [
                {
                    'date': f.date,
                    'high': f.temperature_high,
                    'low': f.temperature_low,
                    'description': f.description,
                    'humidity': f.humidity,
                    'wind_speed': f.wind_speed,
                    'rain_chance': f.precipitation_chance
                } for f in forecast
            ],
            'recommendations': self._generate_weather_recommendations(current, forecast)
        }
    
    def _generate_weather_recommendations(self, current: Dict, forecast: List[WeatherData]) -> List[str]:
        """Generate weather-based travel recommendations"""
        recommendations = []
        
        if current['temperature'] < 10:
            recommendations.append("Pack warm clothing and layers")
        elif current['temperature'] > 30:
            recommendations.append("Bring sun protection and stay hydrated")
        
        if current['humidity'] > 80:
            recommendations.append("High humidity - consider indoor activities during midday")
        
        rainy_days = sum(1 for f in forecast if f.precipitation_chance > 60)
        if rainy_days > 2:
            recommendations.append("Pack an umbrella - several rainy days expected")
        
        temp_variation = max(f.temperature_high for f in forecast) - min(f.temperature_low for f in forecast)
        if temp_variation > 15:
            recommendations.append("Pack for varying temperatures - significant changes expected")
        
        return recommendations