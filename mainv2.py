import asyncio

import dotenv

dotenv.load_dotenv()

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
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

# Import your enhanced ADK agent with citations
from news_agent.agentv2 import (
    root_agent, 
    CompanyProfile
)

# Initialize FastAPI app
app = FastAPI(
    title="Company Data Extraction API with Citations",
    description="Extract comprehensive company data with source citations using Google ADK",
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

class CompanyRequest(BaseModel):
    company_name: str

class CompanyResponse(BaseModel):
    company_name: str
    data: CompanyProfile
    source: str  # "database" or "extraction" or "forced_extraction"
    last_updated: str
    cache_age_days: int
    extraction_status: str


class HealthResponse(BaseModel):
    message: str
    status: str
    timestamp: str

# --- Helper Functions ---

def validate_company_name(company_name: str) -> tuple[bool, str]:
    """Validate company name input"""
    if not company_name or not company_name.strip():
        return False, "Company name is required"
    if len(company_name.strip()) < 2:
        return False, "Company name must be at least 2 characters"
    return True, ""


async def extract_company_data_with_adk(company_name: str) -> CompanyProfile:
    """Extract company data using the enhanced ADK agent with citations"""

    start_time = datetime.now(timezone.utc)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="company_extraction",
        session_service=session_service
    )
    session_id = f"extract_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name="company_extraction",
        user_id="api_user",
        session_id=session_id
    )

    content = types.Content(role="user", parts=[types.Part(text=company_name)])

    print(f"🤖 Starting enhanced ADK extraction with citations for: {company_name}")
    try:
        # Run the agent and collect all events into a list
        events = [
            event for event in runner.run(
                user_id="api_user",
                session_id=session_id,
                new_message=content,
            )
        ]

        # The final result is the very last event
        final_event = events[-1] if events else None

        if (
            final_event
            and final_event.is_final_response()
            and final_event.content
            and final_event.content.parts
        ):
            print("✅ Final JSON event with citations received from agent.")
            json_string = final_event.content.parts[0].text
            
            try:
                final_structured_output = json.loads(json_string)
                company_profile = CompanyProfile(**final_structured_output)
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                print(f"✅ Enhanced ADK extraction completed for: {company_name} ({duration:.1f}s)")
                return company_profile
                
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from agent: {e}")
            except Exception as e:
                raise Exception(f"Failed to parse company profile: {e}")
        else:
            # If for some reason there's no final event, raise an error
            raise Exception("Agent did not produce a valid final response.")

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"❌ Enhanced ADK extraction failed for: {company_name} - {e}")
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
        message="Company Data Extraction API with Citations is running!",
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/extract", response_model=CompanyResponse)
async def extract_company(request: CompanyRequest):
    """
    Main endpoint: Extract company data with citations and Firebase caching
    
    - Checks Firebase cache first
    - Extracts fresh data with citations if cache is stale/missing
    - Returns structured company profile with source citations
    """
    company_name = request.company_name.strip()
    
    # Validate input
    is_valid, error_msg = validate_company_name(company_name)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Extract fresh data using enhanced ADK agent
    print(f"🔍 Extracting fresh data with citations for: {company_name}")
    company_profile = await extract_company_data_with_adk(company_name)
    print(f"✅ Extraction with citations completed for: {company_name}")
    print(f"📝 Company Profile: {company_profile.json()}")
    
    return CompanyResponse(
        company_name=company_name,
        data=company_profile,
        source="extraction",
        last_updated=datetime.now(timezone.utc).isoformat(),
        cache_age_days=0,
        extraction_status="completed"
    )
    
# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)