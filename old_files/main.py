import asyncio
import dotenv

dotenv.load_dotenv()

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
import json

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if not os.getenv('GOOGLE_API_KEY'):
    print("WARNING: GOOGLE_API_KEY not found in environment variables")
    print("Please add GOOGLE_API_KEY to your .env file")

# Import the news aggregation agent
from news_agent.agentv3 import (
    root_agent, 
    CompanyNewsProfile
)

# Initialize FastAPI app
app = FastAPI(
    title="Indian Startup News Aggregation API",
    description="Extract comprehensive news coverage for Indian startups with source citations using Google ADK",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class NewsRequest(BaseModel):
    company_name: str
    years: Optional[int] = 3  # Default to 3 years of news

class NewsResponse(BaseModel):
    company_name: str
    news_data: CompanyNewsProfile
    extraction_timestamp: str
    total_articles_found: int
    years_covered: int
    extraction_status: str

class HealthResponse(BaseModel):
    message: str
    status: str
    timestamp: str
    api_version: str

# --- Helper Functions ---

def validate_company_name(company_name: str) -> tuple[bool, str]:
    """Validate company name input"""
    if not company_name or not company_name.strip():
        return False, "Company name is required"
    if len(company_name.strip()) < 2:
        return False, "Company name must be at least 2 characters"
    return True, ""

def clean_json_string(json_str: str) -> str:
    """Clean and fix common JSON issues"""
    import re
    
    # Remove markdown code blocks
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    # Replace smart quotes with regular quotes
    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace("'", "'").replace("'", "'")
    
    # Fix common escape issues in strings
    # This is a simple approach - find content between quotes and escape internal quotes
    lines = json_str.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that are just structural JSON (brackets, commas)
        if line.strip() in ['{', '}', '[', ']', ',', '{,', '},', '[,', '],']:
            cleaned_lines.append(line)
            continue
            
        # For lines with key-value pairs, protect the content
        if '": "' in line or "': '" in line:
            # Simple protection: if line has unescaped quotes in the value, try to fix
            # This is a heuristic approach
            parts = line.split('": "', 1)
            if len(parts) == 2:
                key_part = parts[0]
                value_part = parts[1]
                
                # Find the closing quote for the value
                # Handle cases where value contains quotes
                if value_part.count('"') > 1:
                    # Remove internal quotes or escape them
                    last_quote = value_part.rfind('"')
                    if last_quote > 0:
                        value_content = value_part[:last_quote]
                        rest = value_part[last_quote:]
                        # Remove quotes from content
                        value_content = value_content.replace('"', '')
                        value_part = value_content + rest
                
                line = key_part + '": "' + value_part
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

async def extract_news_with_adk(company_name: str, years: int = 3) -> CompanyNewsProfile:
    """Extract comprehensive news coverage using the ADK agent"""

    start_time = datetime.now(timezone.utc)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="news_extraction",
        session_service=session_service
    )
    session_id = f"news_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name="news_extraction",
        user_id="api_user",
        session_id=session_id
    )

    # Enhanced prompt for news extraction
    prompt = f"""Extract comprehensive news coverage for the Indian startup: {company_name}

Analysis Period: Last {years} years ({datetime.now().year - years + 1} to {datetime.now().year})

Please find ALL news articles, press releases, and media coverage including:
- Funding announcements and investment news
- Product launches and updates
- Leadership changes and appointments
- Business expansion and partnerships
- Awards and recognition
- Challenges, controversies, or regulatory issues
- Customer milestones and growth metrics

Focus on Indian startup media sources and provide complete citations for every article."""

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    print(f"🔍 Starting news extraction for: {company_name} ({years} years)")
    try:
        # Run the agent and collect all events
        events = [
            event for event in runner.run(
                user_id="api_user",
                session_id=session_id,
                new_message=content,
            )
        ]

        # Get the final result
        final_event = events[-1] if events else None

        if (
            final_event
            and final_event.is_final_response()
            and final_event.content
            and final_event.content.parts
        ):
            print("✅ Final JSON event received from news agent.")
            json_string = final_event.content.parts[0].text
            
            try:
                final_structured_output = json.loads(json_string)
                news_profile = CompanyNewsProfile(**final_structured_output)
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                print(f"✅ News extraction completed for: {company_name} ({duration:.1f}s)")
                print(f"📰 Total articles found: {news_profile.total_articles_found}")
                return news_profile
                
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from agent: {e}")
            except Exception as e:
                raise Exception(f"Failed to parse news profile: {e}")
        else:
            raise Exception("Agent did not produce a valid final response.")

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"❌ News extraction failed for: {company_name} - {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        print(f"🧹 Finished processing for session: {session_id}")
        session_service = None
        print(f"✅ Session cleanup completed for: {session_id}")

# --- API Routes ---

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        message="Indian Startup News Aggregation API is running!",
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        api_version="2.0.0"
    )

@app.post("/extract-news", response_model=NewsResponse)
async def extract_news(request: NewsRequest):
    """
    Main endpoint: Extract comprehensive news coverage for an Indian startup
    
    - Searches across multiple Indian startup media sources
    - Organizes news by year and category
    - Provides complete citations for all articles
    - Includes funding, product, leadership, expansion, and challenge news
    
    Parameters:
    - company_name: Name of the Indian startup
    - years: Number of years to cover (default: 3, range: 1-5)
    """
    company_name = request.company_name.strip()
    years = max(1, min(request.years or 3, 5))  # Limit between 1-5 years
    
    # Validate input
    is_valid, error_msg = validate_company_name(company_name)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Extract news using ADK agent
    print(f"🔍 Extracting {years} years of news for: {company_name}")
    news_profile = await extract_news_with_adk(company_name, years)
    print(f"✅ News extraction completed for: {company_name}")
    print(f"📊 Articles by year: {len(news_profile.news_by_year)} years covered")
    print(f"💰 Funding articles: {news_profile.funding_news.total_funding_articles}")
    print(f"🚀 Product articles: {news_profile.product_news.total_product_articles}")
    
    return NewsResponse(
        company_name=company_name,
        news_data=news_profile,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        total_articles_found=news_profile.total_articles_found,
        years_covered=years,
        extraction_status="completed"
    )

@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy", "service": "news-extraction-api"}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Indian Startup News Aggregation API...")
    print("📡 Server will be available at: http://localhost:5005")
    print("📚 API Documentation: http://localhost:5005/docs")
    uvicorn.run(app, host="0.0.0.0", port=5005)