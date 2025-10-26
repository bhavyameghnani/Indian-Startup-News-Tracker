import asyncio
import dotenv

dotenv.load_dotenv()

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
import json
from pathlib import Path

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
from news_agent.news_agent import (
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Pydantic Models ---

class NewsRequest(BaseModel):
    company_name: str
    years: Optional[int] = 3

class NewsResponse(BaseModel):
    company_name: str
    report_path: str
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

def clean_json_response(json_str: str) -> str:
    """Clean and fix JSON response from agent"""
    import re
    
    # Remove markdown code blocks if present
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    # Try to parse as-is first
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    # Fix common issues
    # Replace smart quotes with regular quotes
    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace("'", "'").replace("'", "'")
    
    # Fix unescaped quotes in strings
    # This regex finds strings and escapes internal quotes
    def fix_string_content(match):
        content = match.group(1)
        # Escape any unescaped quotes within the string
        content = content.replace('\\', '\\\\')  # Escape backslashes first
        content = content.replace('"', '\\"')     # Escape quotes
        return f'"{content}"'
    
    # Try to fix string values between quotes
    try:
        # Pattern to match JSON string values
        # This is a simplified approach - matches "key": "value" patterns
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If line contains a string value
            if '": "' in line:
                # Split carefully to preserve structure
                parts = line.split('": "', 1)
                if len(parts) == 2:
                    key_part = parts[0] + '": "'
                    value_part = parts[1]
                    
                    # Find the end quote, accounting for escaped quotes
                    in_escape = False
                    end_pos = -1
                    for i, char in enumerate(value_part):
                        if char == '\\':
                            in_escape = not in_escape
                        elif char == '"' and not in_escape:
                            end_pos = i
                            break
                        else:
                            in_escape = False
                    
                    if end_pos > 0:
                        value_content = value_part[:end_pos]
                        rest = value_part[end_pos:]
                        
                        # Clean the value content
                        # Remove or escape problematic characters
                        value_content = value_content.replace('\n', ' ')
                        value_content = value_content.replace('\r', ' ')
                        value_content = value_content.replace('\t', ' ')
                        
                        line = key_part + value_content + rest
            
            fixed_lines.append(line)
        
        json_str = '\n'.join(fixed_lines)
    except Exception as e:
        print(f"Warning: Could not fully clean JSON: {e}")
    
    return json_str

def generate_markdown_report(news_profile: CompanyNewsProfile, company_name: str) -> str:
    """Generate a comprehensive Markdown report from the news profile"""
    
    md_content = f"""# News Report: {company_name}

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}  
**Analysis Period:** {news_profile.analysis_period}  
**Total Articles Found:** {news_profile.total_articles_found}  
**Media Sentiment:** {news_profile.media_sentiment}

---

## Executive Summary

{news_profile.overall_trajectory}

### Key Themes
"""
    
    for theme in news_profile.key_themes:
        md_content += f"- {theme}\n"
    
    md_content += "\n### Major Milestones\n\n"
    for i, milestone in enumerate(news_profile.major_milestones, 1):
        md_content += f"{i}. {milestone}\n"
    
    # News by Year
    md_content += "\n---\n\n## News Coverage by Year\n\n"
    
    for yearly_news in sorted(news_profile.news_by_year, key=lambda x: x.year, reverse=True):
        md_content += f"### {yearly_news.year} ({yearly_news.article_count} articles)\n\n"
        md_content += f"**Year Summary:** {yearly_news.year_summary}\n\n"
        
        for article in yearly_news.articles:
            md_content += f"#### {article.headline}\n\n"
            md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
            if article.published_date:
                md_content += f"**Date:** {article.published_date}  \n"
            md_content += f"**Category:** {article.category}\n\n"
            md_content += f"{article.summary}\n\n"
            
            if article.key_points:
                md_content += "**Key Points:**\n"
                for point in article.key_points:
                    md_content += f"- {point}\n"
            md_content += "\n"
    
    # Funding News
    md_content += "\n---\n\n## 💰 Funding News\n\n"
    md_content += f"**Total Funding Articles:** {news_profile.funding_news.total_funding_articles}\n\n"
    md_content += f"**Summary:** {news_profile.funding_news.total_funding_summary}\n\n"
    
    for article in news_profile.funding_news.funding_timeline:
        md_content += f"### {article.headline}\n\n"
        md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
        if article.published_date:
            md_content += f"**Date:** {article.published_date}\n\n"
        md_content += f"{article.summary}\n\n"
    
    # Product News
    md_content += "\n---\n\n## 🚀 Product News\n\n"
    md_content += f"**Total Product Articles:** {news_profile.product_news.total_product_articles}\n\n"
    md_content += f"**Summary:** {news_profile.product_news.product_evolution_summary}\n\n"
    
    for article in news_profile.product_news.product_milestones:
        md_content += f"### {article.headline}\n\n"
        md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
        if article.published_date:
            md_content += f"**Date:** {article.published_date}\n\n"
        md_content += f"{article.summary}\n\n"
    
    # Leadership News
    md_content += "\n---\n\n## 👥 Leadership & Management\n\n"
    md_content += f"**Total Leadership Articles:** {news_profile.leadership_news.total_leadership_articles}\n\n"
    md_content += f"**Summary:** {news_profile.leadership_news.leadership_summary}\n\n"
    
    for article in news_profile.leadership_news.leadership_changes:
        md_content += f"### {article.headline}\n\n"
        md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
        if article.published_date:
            md_content += f"**Date:** {article.published_date}\n\n"
        md_content += f"{article.summary}\n\n"
    
    # Expansion News
    md_content += "\n---\n\n## 🌍 Business Expansion\n\n"
    md_content += f"**Total Expansion Articles:** {news_profile.expansion_news.total_expansion_articles}\n\n"
    md_content += f"**Summary:** {news_profile.expansion_news.expansion_summary}\n\n"
    
    for article in news_profile.expansion_news.expansion_milestones:
        md_content += f"### {article.headline}\n\n"
        md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
        if article.published_date:
            md_content += f"**Date:** {article.published_date}\n\n"
        md_content += f"{article.summary}\n\n"
    
    # Challenges News
    if news_profile.challenges_news.total_challenge_articles > 0:
        md_content += "\n---\n\n## ⚠️ Challenges & Issues\n\n"
        md_content += f"**Total Challenge Articles:** {news_profile.challenges_news.total_challenge_articles}\n\n"
        md_content += f"**Summary:** {news_profile.challenges_news.challenges_summary}\n\n"
        
        for article in news_profile.challenges_news.challenges:
            md_content += f"### {article.headline}\n\n"
            md_content += f"**Source:** [{article.source_name}]({article.source_url})  \n"
            if article.published_date:
                md_content += f"**Date:** {article.published_date}\n\n"
            md_content += f"{article.summary}\n\n"
    
    md_content += "\n---\n\n*Report generated by Indian Startup News Aggregation API*\n"
    
    return md_content

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

Provide complete citations for every article."""

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    print(f"🔍 Starting news extraction for: {company_name} ({years} years)")
    try:
        events = [
            event for event in runner.run(
                user_id="api_user",
                session_id=session_id,
                new_message=content,
            )
        ]

        final_event = events[-1] if events else None

        if (
            final_event
            and final_event.is_final_response()
            and final_event.content
            and final_event.content.parts
        ):
            print("✅ Final JSON event received from news agent.")
            json_string = final_event.content.parts[0].text
            
            # Save raw response for debugging
            debug_path = OUTPUT_DIR / f"debug_response_{uuid.uuid4().hex[:8]}.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"🐛 Raw response saved to: {debug_path}")
            
            # Clean the JSON string before parsing
            json_string = clean_json_response(json_string)
            
            try:
                final_structured_output = json.loads(json_string)
                news_profile = CompanyNewsProfile(**final_structured_output)
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                print(f"✅ News extraction completed for: {company_name} ({duration:.1f}s)")
                print(f"📰 Total articles found: {news_profile.total_articles_found}")
                return news_profile
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"   Error at line {e.lineno}, column {e.colno}")
                
                # Try more aggressive cleaning
                try:
                    # Remove all newlines and extra spaces from string values
                    import re
                    # This is a last-resort fix - extract what we can
                    json_string_compact = re.sub(r'\s+', ' ', json_string)
                    final_structured_output = json.loads(json_string_compact)
                    news_profile = CompanyNewsProfile(**final_structured_output)
                    print("✅ Successfully parsed with aggressive cleaning")
                    return news_profile
                except Exception as e2:
                    print(f"❌ Aggressive cleaning also failed: {e2}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Invalid JSON from agent. Debug file saved to: {debug_path}. Error: {str(e)}"
                    )
                    
            except Exception as e:
                print(f"❌ Failed to parse news profile: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create news profile: {str(e)}. Check debug file: {debug_path}"
                )
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
    - Generates a comprehensive Markdown report
    
    Parameters:
    - company_name: Name of the Indian startup
    - years: Number of years to cover (default: 3, range: 1-5)
    """
    company_name = request.company_name.strip()
    years = max(1, min(request.years or 3, 5))
    
    # Validate input
    is_valid, error_msg = validate_company_name(company_name)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Extract news using ADK agent
    print(f"🔍 Extracting {years} years of news for: {company_name}")
    news_profile = await extract_news_with_adk(company_name, years)
    print(f"✅ News extraction completed for: {company_name}")
    
    # Generate Markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)
    report_filename = f"{safe_company_name}_{timestamp}.md"
    report_path = OUTPUT_DIR / report_filename
    
    markdown_content = generate_markdown_report(news_profile, company_name)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"📝 Report saved to: {report_path}")
    
    return NewsResponse(
        company_name=company_name,
        report_path=str(report_path),
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
    print(f"📁 Reports will be saved to: {OUTPUT_DIR.absolute()}")
    uvicorn.run(app, host="0.0.0.0", port=5005)