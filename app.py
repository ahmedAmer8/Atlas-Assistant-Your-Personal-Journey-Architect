import streamlit as st
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

from agents.atlas_agent import AtlasAgent
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Atlas Assistant - Your Personal Journey Architect",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class AtlasUI:
    """Streamlit UI for Atlas Assistant"""
    
    def __init__(self):
        self.initialize_session_state()
        self.agent = self.get_agent()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'agent_stats' not in st.session_state:
            st.session_state.agent_stats = {}
    
    def get_agent(self) -> AtlasAgent:
        """Get or create Atlas agent"""
        if st.session_state.agent is None:
            with st.spinner("Initializing Atlas Assistant..."):
                try:
                    st.session_state.agent = AtlasAgent()
                    st.success("Atlas Assistant ready!")
                except Exception as e:
                    st.error(f"Failed to initialize Atlas Assistant: {e}")
                    st.stop()
        return st.session_state.agent
    
    def render_sidebar(self):
        """Render sidebar with controls and information"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1f4e79, #2196f3); border-radius: 10px; margin-bottom: 20px;">
                <h1 style="color: white; margin: 0; font-size: 2em;">🗺️ ATLAS</h1>
                <p style="color: #e3f2fd; margin: 5px 0 0 0; font-size: 0.9em;">Your Personal Journey Architect</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("🎛️ Controls")
            
            creativity_level = st.selectbox(
                "Creativity Level",
                ["conservative", "balanced", "creative"],
                index=1,
                help="Adjust how creative vs practical the responses are"
            )
            
            if st.button("Apply Settings"):
                self.agent.set_creativity_level(creativity_level)
                st.success(f"Set to {creativity_level} mode")
            
            if st.button("Reset Session"):
                self.agent.reset_session()
                st.session_state.chat_history = []
                st.success("Session reset!")
                st.rerun()
            
            st.markdown("---")
            
            self.render_stats_sidebar()
            
            st.markdown("---")
            
            
            st.subheader("💡 Tips")
            st.markdown("""
            **Try asking:**
            - "Plan a 3-day trip to Paris"
            - "I love museums and have $500 budget"
            - "What's the weather like in Tokyo?"
            - "Find romantic restaurants in Rome"
            - "Budget-friendly activities in Berlin"
            """)
    
    def render_stats_sidebar(self):
        """Render agent statistics in sidebar"""
        try:
            stats = self.agent.get_statistics()
            st.session_state.agent_stats = stats
            
            st.subheader("📊 Session Stats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", stats.get("total_messages", 0))
            with col2:
                attractions_count = stats.get("place_finder_stats", {}).get("total_attractions", 0)
                st.metric("Attractions", attractions_count)
            
            user_prefs = stats.get("user_preferences")
            if user_prefs:
                st.subheader("👤 Your Preferences")
                
                if user_prefs.get("interests"):
                    st.write("**Interests:**", ", ".join(user_prefs["interests"]))
                
                if user_prefs.get("travel_style", "mid_range") != "mid_range":
                    st.write("**Style:**", user_prefs["travel_style"].title())
                
                if user_prefs.get("budget_range", (0, 1000)) != (0, 1000):
                    budget_range = user_prefs["budget_range"]
                    st.write("**Budget:**", f"${budget_range[0]:.0f} - ${budget_range[1]:.0f}")
        
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    
    def render_main_interface(self):
        """Render main chat interface"""
        st.markdown('<h1 class="main-header">🗺️ Atlas Assistant</h1>', unsafe_allow_html=True)
        st.markdown("*Your Personal Journey Architect - Planning adventures with AI precision*")
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                self.render_chat_message(message)
        
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.chat_input(
                    "Ask me anything about travel planning...",
                    key="chat_input"
                )
            
            with col2:
                if st.button("🎯", help="Quick examples", use_container_width=True):
                    self.show_quick_examples()
        
        if user_input:
            self.process_user_input(user_input)
    
    def render_chat_message(self, message: Dict):
        """Render individual chat message"""
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                if "metadata" in message and message["metadata"]:
                    self.render_tool_results(message["metadata"])
    
    def render_tool_results(self, metadata: Dict):
        """Render tool results in expandable sections"""
        if "weather" in metadata:
            with st.expander("🌤️ Weather Details"):
                self.render_weather_data(metadata["weather"])
        
        if "places" in metadata:
            with st.expander("📍 Attractions Found"):
                self.render_places_data(metadata["places"])
        
        if "budget" in metadata:
            with st.expander("💰 Budget Breakdown"):
                self.render_budget_data(metadata["budget"])
    
    def render_weather_data(self, weather_data: Dict):
        """Render weather information"""
        if "error" in weather_data:
            st.error(weather_data["error"])
            return
        
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", [])
        
        if current:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{current['temperature']:.1f}°C")
            with col2:
                st.metric("Feels Like", f"{current['feels_like']:.1f}°C")
            with col3:
                st.metric("Humidity", f"{current['humidity']}%")
            
            st.write(f"**Condition:** {current['description']}")
        
        if forecast:
            df = pd.DataFrame(forecast)
            fig = px.line(df, x='date', y=['high', 'low'], 
                        title="Temperature Forecast",
                        labels={'value': 'Temperature (°C)', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_places_data(self, places_data: List[Dict]):
        """Render places/attractions information"""
        if not places_data:
            st.info("No attractions found.")
            return
        
        df = pd.DataFrame(places_data)
        
        for i, place in enumerate(places_data[:5]):  # Show top 5
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{place['name']}**")
                    st.write(f"📍 {place['city']}, {place.get('country', '')}")
                    st.write(f"🏷️ {place['category']}")
                
                with col2:
                    st.metric("Cost", f"${place['avg_cost_usd']:.2f}")
                    st.write(f"⭐ {place.get('rating', 'N/A')}")
                
                with col3:
                    similarity = place.get('similarity_score', 0)
                    st.metric("Match", f"{similarity:.1%}")
                
                st.write(place.get('description', '')[:150] + "...")
                st.markdown("---")
    
    def render_budget_data(self, budget_data: Dict):
        """Render budget information"""
        if isinstance(budget_data, dict):
            total_cost = budget_data.get('total_cost', 0)
            final_budget = budget_data.get('final_budget', 0)
            margin = budget_data.get('margin_amount', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Base Cost", f"${total_cost:.2f}")
            with col2:
                st.metric("With Margin", f"${final_budget:.2f}")
            with col3:
                st.metric("Safety Buffer", f"${margin:.2f}")
            
            breakdown = budget_data.get('breakdown', {})
            if breakdown:
                st.subheader("Cost Breakdown")
                for category, cost in breakdown.items():
                    st.write(f"**{category.title()}:** ${cost:.2f}")
    
    def show_quick_examples(self):
        """Show quick example prompts"""
        examples = [
            "Plan a 3-day trip to Paris for $800",
            "I want museums and art galleries in Rome",
            "Budget backpacking in Tokyo for a week",
            "Romantic weekend in Barcelona",
            "Family-friendly activities in London"
        ]
        
        st.sidebar.markdown("### 🎯 Quick Examples")
        for example in examples:
            if st.sidebar.button(example, key=f"example_{hash(example)}"):
                self.process_user_input(example)
    
    def process_user_input(self, user_input: str):
        """Process user input and get agent response"""
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        with st.spinner("Atlas is thinking..."):
            try:
                response = self.agent.process_message(user_input)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "metadata": response.get("metadata", {}),
                    "timestamp": datetime.now()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {e}")
                logger.error(f"Error in process_user_input: {e}")
    
    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_interface()

def main():
    """Main application function"""
    try:
        app = AtlasUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()