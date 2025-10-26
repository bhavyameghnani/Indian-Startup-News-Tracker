"""
LVX Startup Analysis Podcast API v1.0
API for generating startup analysis podcasts for investors
Powered by Let's Venture Platform
"""

import asyncio
import dotenv
import json
import os
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google import genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import agents
from startup_agent import (
    root_agent,
    pdf_analysis_agent,
)

dotenv.load_dotenv()

if not os.getenv('GOOGLE_API_KEY'):
    print("WARNING: GOOGLE_API_KEY not found in environment variables")
    print("Please add GOOGLE_API_KEY to your .env file")

# Create podcasts folder
PODCASTS_FOLDER = Path("startup_podcasts")
PODCASTS_FOLDER.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="LVX Startup Analysis Podcast Generator",
    description="Generate investment-grade startup analysis podcasts for Let's Venture Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class StartupAnalysisRequest(BaseModel):
    """Request to generate startup analysis podcast."""
    startup_name: str = Field(..., description="Indian startup name to analyze (e.g., 'Zoho', 'CRED', 'Razorpay')")


class PodcastResponse(BaseModel):
    """Response with podcast files and metadata."""
    status: str
    message: str
    session_id: str
    startup_name: str
    files: dict
    generated_at: str


class PDFAnalysisResponse(BaseModel):
    """Response for PDF-based startup analysis."""
    status: str
    message: str
    session_id: str
    pdf_filename: str
    files: dict
    generated_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    version: str
    platform: str
    timestamp: str


# --- Helper Functions ---
async def validate_startup_name(startup_name: str) -> Tuple[bool, str]:
    """Validate if this is a legitimate Indian startup using Gemini."""
    try:
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

        prompt = f"""Determine if "{startup_name}" is a legitimate Indian startup or company.

Consider:
- Is it a known Indian startup/company?
- Does it operate in India or was founded by Indians?
- Is it in tech, fintech, e-commerce, SaaS, or other startup sectors?
- Examples of valid: Zoho, CRED, Razorpay, Swiggy, Zomato, Ola, Paytm, Freshworks, etc.

Respond with ONLY JSON:
{{"is_valid_startup": true/false, "reason": "brief explanation", "sector": "industry sector if valid"}}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        response_text = response.text.strip()

        # Clean JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        is_valid = result.get("is_valid_startup", False)
        reason = result.get("reason", "Unable to classify")

        return is_valid, reason

    except Exception as e:
        print(f"Startup validation error: {e}")
        return True, "Startup accepted (validation unavailable)"


async def generate_podcast_with_adk(
    session_id: str,
    startup_name: str,
    session_folder: Path
) -> dict:
    """Generate startup analysis podcast using ADK agent pipeline."""

    start_time = datetime.now(timezone.utc)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="startup_analysis_podcast",
        session_service=session_service
    )

    await session_service.create_session(
        app_name="startup_analysis_podcast",
        user_id="lvx_investor",
        session_id=session_id
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Create investment analysis podcast for Indian startup: {startup_name}\nSession ID: {session_id}")]
    )

    print(f"🚀 Starting startup analysis for: {startup_name}")
    print(f"📁 Session folder: {session_folder}")

    try:
        events = [
            event for event in runner.run(
                user_id="lvx_investor",
                session_id=session_id,
                new_message=content,
            )
        ]

        print(f"✅ Received {len(events)} events from agent")

        # Find final response
        final_event = None
        for event in reversed(events):
            try:
                if getattr(event, 'is_final_response', None) and callable(event.is_final_response) and event.is_final_response():
                    final_event = event
                    break
            except Exception:
                pass

        if not final_event:
            for event in reversed(events):
                if getattr(event, 'content', None) and getattr(event.content, 'parts', None):
                    final_event = event
                    break

        if final_event and getattr(final_event, 'content', None):
            print("✅ Final response received")

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"⏱️  Analysis completed in {duration:.1f}s")

            # Move generated files
            files_created = organize_output_files(session_id, session_folder)

            return {
                "status": "success",
                "files": files_created,
                "duration": duration
            }
        else:
            raise Exception("No valid final response from agent")

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"❌ Analysis failed after {duration:.1f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        session_service = None


def organize_output_files(session_id: str, session_folder: Path) -> dict:
    """Move generated files from root to session folder with proper naming."""

    files_created = {
        "audio_english": None,
        "audio_hindi": None,
        "analysis_report": None,
        "summary_markdown": None,
        "script": None
    }

    # Define expected file patterns
    patterns = {
        "audio_english": f"{session_id}_podcast_english.wav",
        "audio_hindi": f"{session_id}_podcast_hindi.wav",
        "analysis_report": f"{session_id}_analysis.md",
        "summary_markdown": f"{session_id}_summary.md",
        "script": f"{session_id}_script.txt"
    }

    # Check root directory for generated files
    root_files = {
        "audio_english": [
            f"{session_id}_podcast_english.wav",
            "startup_analysis_english.wav",
            "podcast_english.wav"
        ],
        "audio_hindi": [
            f"{session_id}_podcast_hindi.wav",
            "startup_analysis_hindi.wav",
            "podcast_hindi.wav"
        ],
        "analysis_report": [
            "startup_analysis_report.md",
            "analysis_report.md",
            "report.md"
        ],
        "summary_markdown": [
            "podcast_summary.md",
            "summary.md",
            f"{session_id}_summary.md"
        ],
        "script": [
            f"{session_id}_script.txt",
            "script.txt",
            "podcast_script.txt"
        ]
    }

    # Move files to session folder
    for file_type, possible_names in root_files.items():
        for filename in possible_names:
            source = Path(filename)
            if source.exists():
                dest = session_folder / patterns[file_type]
                shutil.move(str(source), str(dest))
                files_created[file_type] = str(dest)
                print(f"✅ Moved {filename} → {dest}")
                break

    return files_created


async def generate_podcast_from_pdf(
    session_id: str,
    pdf_path: Path,
    session_folder: Path
) -> dict:
    """Generate startup analysis from PDF using ADK agent pipeline."""

    start_time = datetime.now(timezone.utc)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=pdf_analysis_agent,
        app_name="pdf_startup_analysis",
        session_service=session_service
    )

    await session_service.create_session(
        app_name="pdf_startup_analysis",
        user_id="lvx_investor",
        session_id=session_id
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Analyze startup document and create investment podcast: {pdf_path}\nSession ID: {session_id}")]
    )

    print(f"📄 Starting PDF startup analysis")
    print(f"📁 Session folder: {session_folder}")

    try:
        events = [
            event for event in runner.run(
                user_id="lvx_investor",
                session_id=session_id,
                new_message=content,
            )
        ]

        print(f"✅ Received {len(events)} events from PDF agent")

        final_event = None
        for event in reversed(events):
            try:
                if getattr(event, 'is_final_response', None) and callable(event.is_final_response) and event.is_final_response():
                    final_event = event
                    break
            except Exception:
                pass

        if not final_event:
            for event in reversed(events):
                if getattr(event, 'content', None) and getattr(event.content, 'parts', None):
                    final_event = event
                    break

        if final_event and getattr(final_event, 'content', None):
            print("✅ Final response received (PDF agent)")

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"⏱️  PDF analysis completed in {duration:.1f}s")

            files_created = organize_output_files(session_id, session_folder)

            return {
                "status": "success",
                "files": files_created,
                "duration": duration
            }
        else:
            raise Exception("No valid final response from PDF agent")

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"❌ PDF analysis failed after {duration:.1f}s: {e}")
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {str(e)}")
    finally:
        session_service = None


# --- API Routes ---
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="LVX Startup Analysis Podcast Generator",
        version="1.0.0",
        platform="Let's Venture",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/analyze-startup", response_model=PodcastResponse)
async def analyze_startup(request: StartupAnalysisRequest):
    """
    Generate investment-grade startup analysis podcast.
    
    Process:
    1. Validates startup name
    2. Researches company overview, products, market position
    3. Analyzes recent news, funding, and hiring activities
    4. Evaluates competitive landscape
    5. Creates comprehensive analysis report
    6. Generates investor-focused podcast script
    7. Creates audio in English and Hindi
    8. Saves all files to /startup_podcasts/{session_id}/
    
    Returns paths to generated files.
    """
    
    session_id = f"lvx_{uuid.uuid4().hex[:8]}"
    session_folder = PODCASTS_FOLDER / session_id
    session_folder.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 NEW STARTUP ANALYSIS REQUEST")
    print(f"Session ID: {session_id}")
    print(f"Startup: {request.startup_name}")
    print(f"Platform: Let's Venture")
    print(f"{'='*70}\n")
    
    try:
        # Step 1: Validate startup
        print("[1/5] Validating startup name...")
        is_valid, reason = await validate_startup_name(request.startup_name)
        
        if not is_valid:
            print(f"❌ Startup rejected: {request.startup_name}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid startup name. '{request.startup_name}' - {reason}"
            )
        
        print(f"✅ Startup validated: {request.startup_name}\n")
        
        # Step 2-5: Generate analysis podcast
        print("[2/5] Researching startup and generating analysis...")
        result = await generate_podcast_with_adk(session_id, request.startup_name, session_folder)
        
        print("[3/5] Analyzing competitive landscape...")
        print("[4/5] Creating investor report...")
        print("[5/5] Generating audio files...\n")
        
        # Prepare response
        response = PodcastResponse(
            status="completed",
            message=f"Successfully generated investment analysis for: {request.startup_name}",
            session_id=session_id,
            startup_name=request.startup_name,
            files={
                "audio_english": result["files"].get("audio_english"),
                "audio_hindi": result["files"].get("audio_hindi"),
                "analysis_report": result["files"].get("analysis_report"),
                "summary_markdown": result["files"].get("summary_markdown"),
                "script": result["files"].get("script"),
                "folder": str(session_folder)
            },
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
        print(f"✅ STARTUP ANALYSIS COMPLETED")
        print(f"📁 Output folder: {session_folder}")
        print(f"⏱️  Duration: {result['duration']:.1f}s\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-from-pdf", response_model=PDFAnalysisResponse)
async def analyze_from_pdf(file: UploadFile = File(...)):
    """
    Generate startup analysis from PDF document.
    
    Process:
    1. Saves uploaded PDF (pitch deck, report, etc.)
    2. Parses PDF using multimodal Gemini
    3. Extracts key business metrics and insights
    4. Enriches with market research
    5. Creates investment analysis report
    6. Generates podcast script
    7. Creates audio in English and Hindi
    
    Returns paths to generated files.
    """
    
    session_id = f"lvx_pdf_{uuid.uuid4().hex[:8]}"
    session_folder = PODCASTS_FOLDER / session_id
    session_folder.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📄 NEW PDF ANALYSIS REQUEST")
    print(f"Session ID: {session_id}")
    print(f"PDF: {file.filename}")
    print(f"Platform: Let's Venture")
    print(f"{'='*70}\n")
    
    try:
        # Step 1: Save PDF
        print("[1/5] Saving uploaded PDF...")
        pdf_path = session_folder / file.filename
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"✅ PDF saved: {pdf_path}\n")
        
        # Step 2-5: Generate analysis
        print("[2/5] Parsing PDF and extracting insights...")
        result = await generate_podcast_from_pdf(session_id, pdf_path, session_folder)
        
        print("[3/5] Researching market context...")
        print("[4/5] Creating investment analysis...")
        print("[5/5] Generating audio files...\n")
        
        # Prepare response
        response = PDFAnalysisResponse(
            status="completed",
            message=f"Successfully analyzed PDF: {file.filename}",
            session_id=session_id,
            pdf_filename=file.filename,
            files={
                "pdf": str(pdf_path),
                "audio_english": result["files"].get("audio_english"),
                "audio_hindi": result["files"].get("audio_hindi"),
                "analysis_report": result["files"].get("analysis_report"),
                "summary_markdown": result["files"].get("summary_markdown"),
                "script": result["files"].get("script"),
                "folder": str(session_folder)
            },
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
        print(f"✅ PDF ANALYSIS COMPLETED")
        print(f"📁 Output folder: {session_folder}")
        print(f"⏱️  Duration: {result['duration']:.1f}s\n")
        
        return response
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcasts/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download generated analysis files."""
    file_path = PODCASTS_FOLDER / session_id / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename} in session {session_id}"
        )
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/analysis-list")
async def list_analyses():
    """List all startup analysis sessions."""
    sessions = {}
    
    if PODCASTS_FOLDER.exists():
        for session_dir in PODCASTS_FOLDER.iterdir():
            if session_dir.is_dir():
                files = {
                    "audio_files": sorted([f.name for f in session_dir.glob("*.wav")]),
                    "reports": sorted([f.name for f in session_dir.glob("*.md")]),
                    "pdfs": sorted([f.name for f in session_dir.glob("*.pdf")]),
                    "scripts": sorted([f.name for f in session_dir.glob("*.txt")])
                }
                
                creation_time = datetime.fromtimestamp(
                    session_dir.stat().st_ctime,
                    tz=timezone.utc
                ).isoformat()
                
                sessions[session_dir.name] = {
                    "files": files,
                    "created_at": creation_time,
                    "total_files": sum(len(v) for v in files.values())
                }
    
    return {
        "status": "success",
        "platform": "Let's Venture",
        "podcasts_folder": str(PODCASTS_FOLDER.absolute()),
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.delete("/analysis/{session_id}")
async def delete_analysis(session_id: str):
    """Delete an analysis session."""
    session_folder = PODCASTS_FOLDER / session_id
    
    if not session_folder.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        shutil.rmtree(session_folder)
        return {
            "status": "success",
            "message": f"Deleted session: {session_id}",
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )


@app.get("/health")
async def detailed_health():
    """Detailed health check with API information."""
    return {
        "status": "healthy",
        "service": "LVX Startup Analysis Podcast Generator",
        "version": "1.0.0",
        "platform": "Let's Venture",
        "description": "Generate investment-grade startup analysis podcasts for Indian startups",
        "endpoints": {
            "health_check": "GET /",
            "analyze_startup": "POST /analyze-startup",
            "analyze_from_pdf": "POST /analyze-from-pdf",
            "download_file": "GET /podcasts/{session_id}/{filename}",
            "list_analyses": "GET /analysis-list",
            "delete_analysis": "DELETE /analysis/{session_id}",
            "detailed_health": "GET /health"
        },
        "features": [
            "Indian startup analysis",
            "Recent news and funding updates",
            "Product launch tracking",
            "Hiring activity monitoring",
            "Competitive landscape analysis",
            "Investment-grade reports",
            "Multi-language audio (English + Hindi)",
            "PDF document analysis"
        ],
        "focus": "Investment decision support for Let's Venture investors",
        "target_audience": "Venture capital investors and startup ecosystem stakeholders",
        "podcasts_folder": str(PODCASTS_FOLDER.absolute()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 LVX STARTUP ANALYSIS PODCAST GENERATOR")
    print("="*70)
    print(f"📡 Server: http://localhost:5007")
    print(f"📚 Docs: http://localhost:5007/docs")
    print(f"📁 Output folder: {PODCASTS_FOLDER.absolute()}")
    print("🎯 Platform: Let's Venture")
    print("🎯 Focus: Investment-grade startup analysis")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5007)