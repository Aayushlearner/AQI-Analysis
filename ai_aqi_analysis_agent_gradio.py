from typing import Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
import gradio as gr
import json
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables from .env file
load_dotenv()

class AQIResponse(BaseModel):
    success: bool
    data: Dict[str, float]
    status: str
    expiresAt: str

class ExtractSchema(BaseModel):
    aqi: float = Field(description="Air Quality Index")
    temperature: float = Field(description="Temperature in degrees Celsius")
    humidity: float = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in kilometers per hour")
    pm25: float = Field(description="Particulate Matter 2.5 micrometers")
    pm10: float = Field(description="Particulate Matter 10 micrometers")
    co: float = Field(description="Carbon Monoxide level")

@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str

class AQIAnalyzer:
    
    def __init__(self) -> None:
        self.firecrawl = FirecrawlApp(api_key=os.getenv("FIRE_CRAWL_API_KEY"))
    
    def _format_url(self, country: str, state: str, city: str) -> str:
        """Format URL based on location, handling cases with and without state"""
        country_clean = country.lower().replace(' ', '-')
        city_clean = city.lower().replace(' ', '-')
        
        if not state or state.lower() == 'none':
            return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"
        
        state_clean = state.lower().replace(' ', '-')
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"
    
    def fetch_aqi_data(self, city: str, state: str, country: str) -> tuple[Dict[str, float], str]:
        """Fetch AQI data using Firecrawl"""
        try:
            url = self._format_url(country, state, city)
            info_msg = f"Accessing URL: {url}"
            
            response = self.firecrawl.extract(
                urls=[f"{url}/*"],
                params={
                    'prompt': 'Extract the current real-time AQI, temperature, humidity, wind speed, PM2.5, PM10, and CO levels from the page. Also extract the timestamp of the data.',
                    'schema': ExtractSchema.model_json_schema()
                }
            )
            
            print("Firecrawl Response:", response)  # Debugging: Print raw response
            
            # Check for None values and assign defaults
            if response['data']['wind_speed'] is None:
                response['data']['wind_speed'] = 0.0  # Default value for wind speed
            
            aqi_response = AQIResponse(**response)
            if not aqi_response.success:
                raise ValueError(f"Failed to fetch AQI data: {aqi_response.status}")
            
            return aqi_response.data, info_msg
            
        except Exception as e:
            error_msg = f"Error fetching AQI data: {str(e)}"
            print(error_msg)  # Debugging: Print error message
            return {
                'aqi': 0,
                'temperature': 0,
                'humidity': 0,
                'wind_speed': 0,
                'pm25': 0,
                'pm10': 0,
                'co': 0
            }, error_msg

class HealthRecommendationAgent:
    
    def __init__(self) -> None:
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def get_recommendations(
        self,
        aqi_data: Dict[str, float],
        user_input: UserInput
    ) -> str:
        prompt = self._create_prompt(aqi_data, user_input)
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        return chat_completion.choices[0].message.content
    
    def _create_prompt(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        return f"""
        Based on the following air quality conditions in {user_input.city}, {user_input.state}, {user_input.country}:
        - Overall AQI: {aqi_data['aqi']}
        - PM2.5 Level: {aqi_data['pm25']} µg/m³
        - PM10 Level: {aqi_data['pm10']} µg/m³
        - CO Level: {aqi_data['co']} ppb
        
        Weather conditions:
        - Temperature: {aqi_data['temperature']}°C
        - Humidity: {aqi_data['humidity']}%
        - Wind Speed: {aqi_data['wind_speed']} km/h
        
        User's Context:
        - Medical Conditions: {user_input.medical_conditions or 'None'}
        - Planned Activity: {user_input.planned_activity}
        **Comprehensive Health Recommendations:**
        1. **Impact of Current Air Quality on Health:**
        2. **Necessary Safety Precautions for Planned Activity:**
        3. **Advisability of Planned Activity:**
        4. **Best Time to Conduct the Activity:**
        """

def analyze_conditions(
    city: str,
    state: str,
    country: str,
    medical_conditions: str,
    planned_activity: str
) -> tuple[str, str, str, str]:
    """Analyze conditions and return AQI data, recommendations, and status messages"""
    try:
        # Initialize analyzers
        aqi_analyzer = AQIAnalyzer()
        health_agent = HealthRecommendationAgent()
        
        # Create user input
        user_input = UserInput(
            city=city,
            state=state,
            country=country,
            medical_conditions=medical_conditions,
            planned_activity=planned_activity
        )
        
        # Get AQI data
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(
            city=user_input.city,
            state=user_input.state,
            country=user_input.country
        )
        
        # Format AQI data for display
        aqi_json = json.dumps({
            "Air Quality Index (AQI)": aqi_data['aqi'],
            "PM2.5": f"{aqi_data['pm25']} µg/m³",
            "PM10": f"{aqi_data['pm10']} µg/m³",
            "Carbon Monoxide (CO)": f"{aqi_data['co']} ppb",
            "Temperature": f"{aqi_data['temperature']}°C",
            "Humidity": f"{aqi_data['humidity']}%",
            "Wind Speed": f"{aqi_data['wind_speed']} km/h"
        }, indent=2)
        
        # Get recommendations
        recommendations = health_agent.get_recommendations(aqi_data, user_input)
        
        warning_msg = """
        ⚠️ Note: The data shown may not match real-time values on the website. 
        This could be due to:
        - Cached data in Firecrawl
        - Rate limiting
        - Website updates not being captured
        
        Consider refreshing or checking the website directly for real-time values.
        """
        
        return aqi_json, recommendations, info_msg, warning_msg
        
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        return "", "Analysis failed", error_msg, ""

def create_demo() -> gr.Blocks:
    """Create and configure the Gradio interface"""
    with gr.Blocks(title="AQI Analysis Agent") as demo:
        gr.Markdown(
            """
            # 🌍 AQI Analysis Agent
            Get personalized health recommendations based on air quality conditions.
            """
        )
        
        # Location Details
        with gr.Row():
            with gr.Column():
                city = gr.Textbox(label="City", placeholder="e.g., Mumbai")
                state = gr.Textbox(
                    label="State",
                    placeholder="Leave blank for Union Territories or US cities",
                    value=""
                )
                country = gr.Textbox(label="Country", value="India")
        
        # Personal Details
        with gr.Row():
            with gr.Column():
                medical_conditions = gr.Textbox(
                    label="Medical Conditions (optional)",
                    placeholder="e.g., asthma, allergies",
                    lines=2
                )
                planned_activity = gr.Textbox(
                    label="Planned Activity",
                    placeholder="e.g., morning jog for 2 hours",
                    lines=2
                )
        
        # Status Messages
        info_box = gr.Textbox(label="ℹ️ Status", interactive=False)
        warning_box = gr.Textbox(label="⚠️ Warning", interactive=False)
        
        # Output Areas
        aqi_data_json = gr.JSON(label="📊 Current Air Quality Data")
        recommendations = gr.Markdown(label="🏥 Health Recommendations")
        
        # Analyze Button
        analyze_btn = gr.Button("🔍 Analyze & Get Recommendations", variant="primary")
        analyze_btn.click(
            fn=analyze_conditions,
            inputs=[
                city,
                state,
                country,
                medical_conditions,
                planned_activity
            ],
            outputs=[aqi_data_json, recommendations, info_box, warning_box]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Mumbai", "Maharashtra", "India", "asthma", "morning walk for 30 minutes"],
                ["Delhi", "", "India", "", "outdoor yoga session"],
                ["New York", "", "United States", "allergies", "afternoon run"],
                ["Kakinada", "Andhra Pradesh", "India", "none", "Tennis for 2 hours"]
            ],
            inputs=[city, state, country, medical_conditions, planned_activity]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)