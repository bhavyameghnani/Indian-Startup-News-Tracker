# agentv4.py

from typing import Dict, List
import pathlib
import wave
import re
from urllib.parse import urlparse
import base64

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search, ToolContext
from google import genai
from google.genai import types
import yfinance as yf
from pydantic import BaseModel, Field

class NewsStory(BaseModel):
    """A single news story with its context."""
    company: str = Field(description="Company name associated with the story (e.g., 'Nvidia', 'OpenAI'). Use 'N/A' if not applicable.")
    ticker: str = Field(description="Stock ticker for the company (e.g., 'NVDA'). Use 'N/A' if private or not found.")
    summary: str = Field(description="A brief, one-sentence summary of the news story.")
    why_it_matters: str = Field(description="A concise explanation of the story's significance or impact.")
    financial_context: str = Field(description="Current stock price and change, e.g., '$950.00 (+1.5%)'. Use 'No financial data' if not applicable.")
    source_domain: str = Field(description="The source domain of the news, e.g., 'techcrunch.com'.")
    process_log: str = Field(description="populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output." ) 

class AINewsReport(BaseModel):
    """A structured report of the latest finance news."""
    title: str = Field(default="Finance Research Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings in the report.")
    stories: List[NewsStory] = Field(description="A list of the individual news stories found.")

class PDFParsedContent(BaseModel):
    """Content extracted from PDF using multimodal Gemini."""
    title: str = Field(description="Title extracted from the PDF document.")
    main_topics: List[str] = Field(description="Key topics and sections from the PDF.")
    key_insights: List[str] = Field(description="Important insights and recommendations from the PDF.")
    companies_mentioned: List[str] = Field(description="Companies, tickers, and entities mentioned in the PDF.")
    portfolio_highlights: str = Field(description="Summary of portfolio allocation, performance, and highlights.")

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Helper function to save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


async def parse_pdf_multimodal(pdf_path: str, tool_context: ToolContext) -> Dict:
    """
    Parse a PDF document using Gemini's multimodal capabilities.
    Extracts wealth management insights, portfolio details, and key recommendations.
    
    Args:
        pdf_path: Path to the PDF file
        tool_context: The ADK tool context
        
    Returns:
        Dictionary with extracted PDF content
    """
    try:
        # Read and encode the PDF file
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.standard_b64encode(pdf_file.read()).decode("utf-8")
        
        client = genai.Client()
        
        prompt = """You are an expert wealth management analyst. Analyze this Wealth Management report document and extract:

1. **Main Title & Document Type**: What is the title and type of report?
2. **Key Topics & Sections**: List all major sections and topics covered
3. **Key Insights & Recommendations**: What are the main recommendations and insights?
4. **Portfolio Highlights**: What are the portfolio allocations, performance metrics, and asset classes mentioned?
5. **Companies & Entities**: Which companies, stocks, sectors, or investment vehicles are mentioned?

Return ONLY a JSON object with these keys:
{
    "title": "string",
    "main_topics": ["list", "of", "topics"],
    "key_insights": ["list", "of", "insights"],
    "companies_mentioned": ["list", "of", "companies/tickers"],
    "portfolio_highlights": "summary of portfolio information"
}

Do not include any other text or explanation."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_data
                            )
                        )
                    ]
                )
            ]
        )
        
        response_text = response.text.strip()
        
        # Parse JSON response - handle potential formatting issues
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        parsed_content = eval(response_text)
        
        return {
            "status": "success",
            "content": parsed_content,
            "message": f"Successfully parsed PDF: {pdf_path}"
        }
        
    except Exception as e:
        error_msg = str(e)[:200]
        return {
            "status": "error",
            "message": f"PDF parsing failed: {error_msg}",
            "content": None
        }


async def generate_podcast_audio(podcast_script: str, tool_context: ToolContext, filename: str = "finance_podcast") -> Dict[str, str]:
    """
    Generates audio from a podcast script in both English and Japanese using Gemini API and saves them as WAV files.

    Args:
        podcast_script: The conversational script to be converted to audio (in English).
        tool_context: The ADK tool context.
        filename: Base filename for the audio files (without extension).

    Returns:
        Dictionary with status and file information for both audio files.
    """
    try:
        client = genai.Client()
        
        # --- Generate English Audio ---
        english_prompt = f"TTS the following conversation between Joe and Jane:\n\n{podcast_script}"
        
        english_response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=english_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(speaker='Joe', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'))),
                            types.SpeakerVoiceConfig(speaker='Jane', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')))
                        ]
                    )
                )
            )
        )
        
        english_data = english_response.candidates[0].content.parts[0].inline_data.data
        
        # Save English audio
        english_filename = f"{filename}_english.wav"
        current_directory = pathlib.Path.cwd()
        english_file_path = current_directory / english_filename
        wave_file(str(english_file_path), english_data)
        
        # --- Translate to Japanese ---
        translation_prompt = f"""Translate the following English podcast conversation to natural Japanese. 
Maintain the speaker labels (Joe: and Jane:) and preserve the conversational tone.

{podcast_script}

Provide ONLY the translated conversation with speaker labels, no additional text."""
        
        translation_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=translation_prompt
        )
        
        japanese_script = translation_response.text.strip()
        
        # --- Generate Japanese Audio ---
        japanese_prompt = f"TTS the following conversation between Joe and Jane:\n\n{japanese_script}"
        
        japanese_response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=japanese_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(speaker='Joe', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Charon'))),
                            types.SpeakerVoiceConfig(speaker='Jane', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Aoede')))
                        ]
                    )
                )
            )
        )
        
        japanese_data = japanese_response.candidates[0].content.parts[0].inline_data.data
        
        # Save Japanese audio
        japanese_filename = f"{filename}_japanese.wav"
        japanese_file_path = current_directory / japanese_filename
        wave_file(str(japanese_file_path), japanese_data)
        
        return {
            "status": "success",
            "message": f"Successfully generated English and Japanese podcast audio files",
            "english_file_path": str(english_file_path.resolve()),
            "english_file_size": len(english_data),
            "japanese_file_path": str(japanese_file_path.resolve()),
            "japanese_file_size": len(japanese_data),
            "japanese_script": japanese_script
        }

    except Exception as e:
        error_msg = str(e)[:200]
        return {"status": "error", "message": f"Audio generation failed: {error_msg}"}
    
def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers.
    """
    financial_data: Dict[str, str] = {}

    valid_tickers = [ticker.upper().strip() for ticker in tickers 
                    if ticker and ticker.upper() not in ['N/A', 'NA', '']]
    
    if not valid_tickers:
        return {ticker: "No financial data" for ticker in tickers}
        
    for ticker_symbol in valid_tickers:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")
            
            if price is not None and change_percent is not None:
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                financial_data[ticker_symbol] = "Price data not available."
        except Exception:
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"
            
    return financial_data

def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}

WHITELIST_DOMAINS = ["techcrunch.com", "venturebeat.com", "theverge.com", "technologyreview.com", "arstechnica.com", "cnbc.com", "bloomberg.com", "reuters.com", "marketwatch.com", "investor.com"]

def filter_news_sources_callback(tool, args, tool_context):
    """Callback to enforce that google_search queries only use whitelisted domains."""
    if tool.name == "google_search":
        original_query = args.get("query", "")
        if any(f"site:{domain}" in original_query.lower() for domain in WHITELIST_DOMAINS):
            return None
        whitelist_query_part = " OR ".join([f"site:{domain}" for domain in WHITELIST_DOMAINS])
        args['query'] = f"{original_query} {whitelist_query_part}"
        print(f"MODIFIED query to enforce whitelist: '{args['query']}'")
    return None

def enforce_data_freshness_callback(tool, args, tool_context):
    """Callback to add a time filter to search queries to get recent news."""
    if tool.name == "google_search":
        query = args.get("query", "")
        if "tbs=qdr:w" not in query:
            args['query'] = f"{query} tbs=qdr:w"
            print(f"MODIFIED query for freshness: '{args['query']}'")
    return None

def initialize_process_log(tool_context: ToolContext):
    """Helper to ensure the process_log list exists in the state."""
    if 'process_log' not in tool_context.state:
        tool_context.state['process_log'] = []

def inject_process_log_after_search(tool, args, tool_context, tool_response):
    """
    Callback: After a successful search, this injects the process_log into the response
    and adds a specific note about which domains were sourced.
    """
    if tool.name == "google_search" and isinstance(tool_response, str):
        urls = re.findall(r'https?://[^\s/]+', tool_response)
        unique_domains = sorted(list(set(urlparse(url).netloc for url in urls)))
        
        if unique_domains:
            sourcing_log = f"Action: Sourced news from the following domains: {', '.join(unique_domains)}."
            current_log = tool_context.state.get('process_log', [])
            tool_context.state['process_log'] = [sourcing_log] + current_log

        final_log = tool_context.state.get('process_log', [])
        print(f"CALLBACK LOG: Injecting process log into tool response: {final_log}")
        return {
            "search_results": tool_response,
            "process_log": final_log
        }
    return tool_response


podcaster_agent = Agent(
    name="podcaster_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are an Audio Generation Specialist. Your single task is to take a provided text script
    and convert it into multi-speaker audio files (English and Japanese) using the `generate_podcast_audio` tool.

    Workflow:
    1. Receive the text script from the user or another agent.
    2. Immediately call the `generate_podcast_audio` tool with the provided script.
    3. Report the result of the audio generation back to the user, mentioning both English and Japanese files were created.
    """,
    tools=[generate_podcast_audio],
)

pdf_document_parser_agent = Agent(
    name="pdf_document_parser",
    model="gemini-2.5-flash",
    instruction="""
    You are a Document Parser Specialist. Your task is to extract comprehensive information from
    Wealth Management PDF reports using multimodal analysis.

    Workflow:
    1. Receive the PDF file path from the calling agent.
    2. Call the `parse_pdf_multimodal` tool to extract structured information.
    3. Return the parsed content for use in podcast generation.
    """,
    tools=[parse_pdf_multimodal],
)

root_agent = Agent(
    name="finance_news_researcher",
    model="gemini-2.5-flash", 
    instruction="""
    **Your Core Identity:**
    You are a Finance News Podcast Producer. Your job is to orchestrate a complete workflow: find the latest finance news on a specific user-provided topic, compile a report, write a script, and generate podcast audio files in English and Japanese, all while keeping the user informed.

    **Topic Information:**
    The user has provided a finance topic for research. Your research scope is strictly limited to this specific topic only. All search queries and analysis must be focused on gathering information relevant to this topic.

    **Crucial Rules:**
    1.  **Resilience is Key:** If you encounter an error or cannot find specific information for one item, you MUST NOT halt the entire process. Use placeholder values like "Not Available", and continue to the next step. Your primary goal is to deliver the final report and podcasts, even if some data points are missing.
    2.  **Topic-Specific Focus:** Your research is strictly limited to the user-specified finance topic. Do not deviate to other topics.
    3.  **User-Facing Communication:** Your interaction has only two user-facing messages: the initial acknowledgment and the final confirmation. All complex work must happen silently in the background between these two messages.

    **Understanding Callback-Modified Tool Outputs:**
    The `google_search` tool is enhanced by callbacks. Its final output is a JSON object with two keys:
    1.  `search_results`: A string containing the actual search results.
    2.  `process_log`: A list of strings describing the filtering actions performed.

    **Required Conversational Workflow:**
    1.  **Acknowledge and Inform:** The VERY FIRST thing you do is respond to the user with: "Okay, I'll start researching the latest finance news on your topic. I will enrich the findings with financial data where available and compile a report for you. This might take a moment."
    2.  **Search (Background Step):** Immediately after acknowledging, use the `google_search` tool to find relevant news. Your query must be specifically tailored to find recent news about the user's finance topic.
    3.  **Analyze & Extract Information (Internal Step):** Process search results to identify key companies, entities, and their relevant details. If financial data cannot be found, use 'N/A'.
    4.  **Get Financial Data (Background Step):** Call the `get_financial_context` tool with any extracted tickers. If the tool returns "Not Available" for any ticker, accept this and proceed. Do not stop or report an error.
    5.  **Structure the Report (Internal Step):** Use the `AINewsReport` schema to structure all gathered information. If financial data was not found, you MUST use "Not Available" in the `financial_context` field. You MUST also populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output.
    6.  **Format for Markdown (Internal Step):** Convert the structured `AINewsReport` data into a well-formatted Markdown string. This MUST include a section at the end called "## Data Sourcing Notes" where you list the items from the `process_log`.
    7.  **Save the Report (Background Step):** Save the Markdown string using `save_news_to_markdown` with the filename `finance_research_report.md`.
    8.  **Create Podcast Script (Internal Step):** After saving the report, you MUST convert the structured `AINewsReport` data into a natural, conversational podcast script between two hosts, 'Joe' (enthusiastic) and 'Jane' (analytical).
    9.  **Generate Audio (Background Step):** Call the `podcaster_agent` tool, passing the complete conversational script you just created to it. This will generate both English and Japanese audio files.
    10. **Final Confirmation:** After the audio is successfully generated, your final response to the user MUST be: "All done. I've compiled the research report, saved it to `finance_research_report.md`, and generated the podcast audio files in English and Japanese for you."
    """,
    tools=[
        google_search,
        get_financial_context,
        save_news_to_markdown,
        AgentTool(agent=podcaster_agent) 
    ],
    # REMOVED output_schema to fix the agent transfer conflict
    before_tool_callback=[
        filter_news_sources_callback,
        enforce_data_freshness_callback,
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)

pdf_podcast_agent = Agent(
    name="wealth_management_podcast_producer",
    model="gemini-2.5-flash",
    instruction="""
    **Your Core Identity:**
    You are a Wealth Management Podcast Producer. Your job is to transform Wealth Management PDF reports into engaging podcasts
    by extracting insights, supplementing with current market data, and creating a professional conversational format in both English and Japanese.

    **Workflow:**
    1. **Acknowledge:** Respond with "I'm analyzing your Wealth Management report and gathering supplementary market data. This will take a moment..."
    
    2. **Parse PDF (Background):** Call the `pdf_document_parser_agent` tool to extract comprehensive information from the PDF using multimodal analysis.
    
    3. **Extract & Analyze (Internal):** Process the parsed PDF content to identify:
       - Key portfolio themes and recommendations
       - Companies and tickers mentioned
       - Asset allocation and performance metrics
       - Investment strategies and insights
    
    4. **Fetch Supplementary Data (Background):** 
       - Call `get_financial_context` with extracted tickers to get current prices
       - Call `google_search` to find recent news about mentioned companies/sectors
    
    5. **Enrich Content (Internal):** Combine PDF insights with current market data and news to create a comprehensive narrative.
    
    6. **Create Markdown Report (Background):** Format all information into a markdown file using `save_news_to_markdown`.
    
    7. **Generate Podcast Script (Internal):** Convert the enriched content into a natural conversational podcast script between two hosts:
       - Joe (enthusiastic wealth advisor) discusses opportunities
       - Jane (analytical strategist) provides critical perspectives
       - Include portfolio highlights, current market context, and forward-looking recommendations
    
    8. **Generate Audio (Background):** Call the `podcaster_agent` tool with the podcast script. This will generate both English and Japanese audio files.
    
    9. **Final Confirmation:** Respond with "Done! I've created your Wealth Management podcast from the PDF report in both English and Japanese. The podcasts include current market data and recent news about mentioned holdings. Files have been saved to the podcast folder."
    """,
    tools=[
        AgentTool(agent=pdf_document_parser_agent),
        get_financial_context,
        google_search,
        save_news_to_markdown,
        AgentTool(agent=podcaster_agent)
    ],
    # REMOVED output_schema to fix the agent transfer conflict
    before_tool_callback=[
        filter_news_sources_callback,
        enforce_data_freshness_callback,
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)