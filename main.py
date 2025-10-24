import asyncio
import dotenv
import json
import re
import os
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from news_agent.agent import root_agent, AINewsReport
from google import genai

dotenv.load_dotenv()

if not os.getenv('GOOGLE_API_KEY'):
    print("WARNING: GOOGLE_API_KEY not found in environment variables")
    print("Please add GOOGLE_API_KEY to your .env file")

# Initialize FastAPI app
app = FastAPI(
    title="Finance News Report Generator API",
    description="Generate 6-month finance news reports on user-specified topics using Google ADK",
    version="2.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Pydantic Models ---
class NewsReportRequest(BaseModel):
    """Request to generate a news report"""
    topic: str = Field(..., description="The finance-related topic (e.g., company name like 'Zoho') to research and create a 6-month news report about")

class NewsReportResponse(BaseModel):
    """Response containing news report and file information"""
    report: AINewsReport
    markdown_file: str
    status: str
    generated_at: str
    session_id: str
    topic: str
    total_urls: int = Field(description="Total number of source URLs found")

class ErrorResponse(BaseModel):
    """Error response"""
    status: str
    message: str
    timestamp: str

# --- Helper Functions ---
async def validate_finance_topic(topic: str) -> tuple[bool, str]:
    """
    Validates if the given topic is finance-related using Gemini API.
    
    Args:
        topic: The topic to validate
        
    Returns:
        tuple: (is_valid, explanation)
    """
    try:
        # Use sync client to avoid async context issues
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
        prompt = f"""You are a topic classifier. Determine if the following topic is finance-related.
        
Topic: "{topic}"

Finance-related topics include but are not limited to: 
- Markets & Trading: stocks, cryptocurrencies, bonds, commodities, forex, derivatives, options, futures
- Investments: investment strategies, portfolio management, mutual funds, ETFs, venture capital, private equity
- Banking & Credit: banking services, credit cards, loans, mortgages, credit unions, alternative lending
- Economics: inflation, interest rates, economic indicators, GDP, unemployment, fiscal policy, monetary policy
- Corporate Finance: mergers & acquisitions, IPOs, corporate earnings, dividends, stock splits, company valuations
- Fintech & Blockchain: cryptocurrency, blockchain technology, digital banking, financial apps, payment systems
- Insurance: life insurance, health insurance, property insurance, claims, underwriting
- Real Estate: property investment, real estate markets, mortgages, REITs, property development
- Personal Finance: budgeting, savings, retirement planning, financial literacy, wealth management
- Companies & Startups: Any company name (e.g., 'Zoho', 'Apple', 'Tesla') is finance-related
- Other Finance: economic news, market analysis, financial regulations, banking industry news

Respond with ONLY a JSON object in this format:
{{"is_finance": true/false, "reason": "brief explanation"}}

Do not include any other text."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        response_text = response.text.strip()
        print(f"Validation response: {response_text}")
        
        # Parse JSON response - handle potential formatting issues
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        is_finance = result.get("is_finance", False)
        reason = result.get("reason", "Unable to classify topic")
        
        print(f"Topic validation result - Is Finance: {is_finance}, Reason: {reason}")
        return is_finance, reason
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error during topic validation: {e}")
        print(f"Response text was: {response_text}")
        return True, "Topic accepted (validation response parsing issue)"
    except Exception as e:
        print(f"Error validating topic: {e}")
        return True, "Topic accepted (validation service temporarily unavailable)"


async def generate_news_report_with_adk(session_id: str, topic: str) -> AINewsReport:
    """
    Generate 6-month news report using the ADK agent for a specific finance topic.

    Args:
        session_id: Unique session identifier
        topic: The finance topic to research (e.g., company name)
        
    Returns:
        AINewsReport: The generated report with all source URLs
    """
    
    start_time = datetime.now(timezone.utc)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="news_report_generation",
        session_service=session_service
    )

    await session_service.create_session(
        app_name="news_report_generation",
        user_id="api_user",
        session_id=session_id
    )

    content = types.Content(
        role="user", 
        parts=[types.Part(text=f"Generate a comprehensive 6-month finance news report about: {topic}. Make sure to include ALL source URLs in a numbered list under '## Source URLs' section.")]
    )

    print(f"Starting ADK news report generation for session: {session_id}, topic: {topic}")
    try:
        events = [
            event for event in runner.run(
                user_id="api_user",
                session_id=session_id,
                new_message=content,
            )
        ]

        print(f"Total events received: {len(events)}")
        
        # Find the final response event
        final_event = None
        for event in reversed(events):
            print(f"Event type: {type(event)}, has is_final_response: {hasattr(event, 'is_final_response')}")
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_event = event
                break
        
        if not final_event:
            for event in reversed(events):
                if hasattr(event, 'content') and event.content and event.content.parts:
                    final_event = event
                    break

        if final_event and hasattr(final_event, 'content') and final_event.content and final_event.content.parts:
            print("Final response received from agent")
            response_text = final_event.content.parts[0].text
            print(f"Agent response length: {len(response_text)}")
            print(f"Agent response preview (first 500 chars):\n{response_text[:500]}\n")
            
            # Try to extract URLs from the generated markdown report
            # Use a more comprehensive URL pattern
            url_pattern = r'https?://[^\s\)\]\<\>\"\']+[^\s\)\]\<\>\"\'\.,;]'
            found_urls = re.findall(url_pattern, response_text)
            
            # Clean URLs (remove trailing punctuation)
            cleaned_urls = []
            for url in found_urls:
                url = re.sub(r'[,\.\)\]\>]+$', '', url)
                cleaned_urls.append(url)
            
            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(cleaned_urls))
            
            print(f"Extracted {len(unique_urls)} URLs from the report")
            if unique_urls:
                print("Sample URLs extracted:")
                for i, url in enumerate(unique_urls[:5], 1):
                    print(f"  {i}. {url}")
            else:
                print("WARNING: No URLs found in the agent's response!")
                print("Checking if '## Source URLs' section exists in response...")
                if "## Source URLs" in response_text or "## source urls" in response_text.lower():
                    print("✓ URL section found in response")
                else:
                    print("✗ URL section NOT found in response")
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"ADK news report generation completed ({duration:.1f}s)")
            
            # Create news report with extracted URLs
            news_report = AINewsReport(
                title=f"Finance Research Report: {topic} (Last 6 Months)",
                report_summary=response_text,
                stories=[],
                all_source_urls=unique_urls
            )
            return news_report
                
        else:
            error_msg = f"No valid final event found. Total events: {len(events)}"
            print(error_msg)
            raise Exception(error_msg)

    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"ADK report generation failed ({duration:.1f}s) - {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    finally:
        print(f"Session cleanup for: {session_id}")
        session_service = None



# --- API Routes ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Finance News Report Generator API",
        "version": "2.2.0",
        "description": "Generate 6-month finance news reports with comprehensive URL tracking",
        "endpoints": {
            "POST /generate": "Generate a new finance news report",
            "GET /download/{filename}": "Download a generated report",
            "GET /list-reports": "List all generated reports",
            "GET /debug-report/{filename}": "Debug and analyze a report file"
        }
    }


@app.post("/generate", response_model=NewsReportResponse)
async def generate_news_report(request: NewsReportRequest):
    """
    Generate 6-month finance news report for a specific topic (e.g., company like Zoho)
    
    Request body:
    - topic: str - The finance-related topic to research (e.g., "Zoho")
    
    Returns:
    - Reports are saved to `output/finance_research_report.md`
    - Includes comprehensive summary of 6 months of news
    - Contains all source URLs

    Process:
    - Validates if topic is finance-related
    - Searches for relevant finance news from last 6 months
    - Compiles structured report with all URLs
    - Saves to output folder
    - Returns report with file references
    """
    session_id = f"report_{uuid.uuid4().hex[:8]}"
    
    print(f"\n{'='*60}")
    print(f"Generating 6-month report for session: {session_id}")
    print(f"Topic: {request.topic}")
    print(f"{'='*60}\n")
    
    try:
        # Validate if topic is finance-related
        print(f"[1/6] Validating topic: '{request.topic}'...")
        is_finance, reason = await validate_finance_topic(request.topic)
        
        if not is_finance:
            print(f"[ERROR] Topic rejected: {request.topic}")
            print(f"[ERROR] Reason: {reason}\n")
            raise HTTPException(
                status_code=400, 
                detail=f"Thanks, but we can only generate reports on finance-related topics. '{request.topic}' doesn't appear to be finance-related. Reason: {reason}"
            )
        
        print(f"[✓] Topic validated successfully: {request.topic}\n")
        
        # Generate report for the validated topic
        print(f"[2/6] Generating 6-month report with ADK agent...")
        report = await generate_news_report_with_adk(session_id, request.topic)
        print(f"[✓] Report generated successfully\n")
        
        print(f"[3/6] Compiling response...")
        print(f"[4/6] Verifying output folder...")
        
        # Verify the report was saved to output folder
        report_path = OUTPUT_DIR / "finance_research_report.md"
        if report_path.exists():
            print(f"[✓] Report saved to: {report_path}")
            
            # Read the file to verify URLs are present
            report_content = report_path.read_text(encoding='utf-8')
            if "## Source URLs" in report_content:
                print(f"[✓] '## Source URLs' section found in report")
            else:
                print(f"[WARNING] '## Source URLs' section NOT found in report")
        else:
            print(f"[WARNING] Report file not found at expected location: {report_path}")
        
        print(f"[5/6] Counting source URLs...")
        total_urls = len(report.all_source_urls)
        print(f"[✓] Total URLs found: {total_urls}")
        
        if total_urls > 0:
            print(f"[✓] Sample URLs:")
            for i, url in enumerate(report.all_source_urls[:3], 1):
                print(f"    {i}. {url}")
        else:
            print(f"[WARNING] No URLs were extracted from the report!")
        print()
        
        # Update response to include URL count
        response = NewsReportResponse(
            report=report,
            markdown_file="output/finance_research_report.md",
            status="completed",
            generated_at=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            topic=request.topic,
            total_urls=total_urls
        )
        
        print(f"[6/6] Report generation completed successfully!")
        print(f"{'='*60}")
        print(f"Session ID: {session_id}")
        print(f"Output file: output/finance_research_report.md")
        print(f"Total URLs: {total_urls}")
        print(f"{'='*60}\n")
        
        return response
        
    except HTTPException as http_err:
        print(f"[HTTPException] {http_err.detail}\n")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_report(filename: str):
    """
    Download a generated report from the output folder
    
    Args:
        filename: Name of the file to download (e.g., 'finance_research_report.md')
        
    Returns:
        FileResponse: The requested file
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in output folder")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"'{filename}' is not a file")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/markdown'
    )


@app.get("/list-reports")
async def list_reports():
    """
    List all reports in the output folder
    
    Returns:
        List of report files with metadata
    """
    try:
        reports = []
        for file_path in OUTPUT_DIR.glob("*.md"):
            stat = file_path.stat()
            
            # Try to count URLs in the file
            try:
                content = file_path.read_text(encoding='utf-8')
                url_pattern = r'https?://[^\s\)\]\<\>\"\']+[^\s\)\]\<\>\"\'\.,;]'
                urls = re.findall(url_pattern, content)
                url_count = len(set(urls))
            except:
                url_count = 0
            
            reports.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "url_count": url_count,
                "download_url": f"/download/{file_path.name}",
                "debug_url": f"/debug-report/{file_path.name}"
            })
        
        return {
            "status": "success",
            "total_reports": len(reports),
            "reports": sorted(reports, key=lambda x: x["modified"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@app.get("/debug-report/{filename}")
async def debug_report(filename: str):
    """
    Debug endpoint to extract and show URLs from a report file
    
    Args:
        filename: Name of the report file to analyze
        
    Returns:
        Detailed analysis of the report including extracted URLs
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Extract URLs using multiple patterns
        url_pattern = r'https?://[^\s\)\]\<\>\"\']+[^\s\)\]\<\>\"\'\.,;]'
        found_urls = re.findall(url_pattern, content)
        cleaned_urls = [re.sub(r'[,\.\)\]\>]+$', '', url) for url in found_urls]
        unique_urls = list(dict.fromkeys(cleaned_urls))
        
        # Check for "Source URLs" section
        has_url_section = "## Source URLs" in content or "## source urls" in content.lower()
        
        # Extract domains
        domains = list(set(urlparse(url).netloc for url in unique_urls if url))
        
        # Count lines in the report
        line_count = content.count('\n')
        
        # Check for specific sections
        sections = {
            "has_title": content.startswith("# ") or "# Finance Research Report" in content,
            "has_summary": "## Report Summary" in content or "## report summary" in content.lower(),
            "has_data_sourcing": "## Data Sourcing" in content or "## data sourcing" in content.lower(),
            "has_source_urls": has_url_section
        }
        
        return {
            "status": "success",
            "filename": filename,
            "file_size_bytes": len(content),
            "line_count": line_count,
            "total_urls_found": len(unique_urls),
            "unique_domains": sorted(domains),
            "sections_present": sections,
            "urls": unique_urls[:20],  # First 20 URLs
            "total_url_count": len(unique_urls),
            "report_preview": content[:1000] + "..." if len(content) > 1000 else content
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze report: {str(e)}")


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Finance News Report Generator API")
    print("="*60)
    print(f"Server starting on: http://0.0.0.0:5006")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5006)