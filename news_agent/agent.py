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
    source_url: str = Field(description="The full URL of the news article.")
    process_log: str = Field(description="populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output.") 

class AINewsReport(BaseModel):
    """A structured report of the latest finance news."""
    title: str = Field(default="Finance Research Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings in the report.")
    stories: List[NewsStory] = Field(description="A list of the individual news stories found.")
    all_source_urls: List[str] = Field(default_factory=list, description="Complete list of all source URLs used in the report.")


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
    Saves the given content to a Markdown file in the 'output' directory.
    Creates the output directory if it doesn't exist.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        
        # Create output directory if it doesn't exist
        current_directory = pathlib.Path.cwd()
        output_directory = current_directory / "output"
        output_directory.mkdir(exist_ok=True)
        
        file_path = output_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}

# Extended whitelist with Indian startup news domains
WHITELIST_DOMAINS = [
    "techcrunch.com", 
    "venturebeat.com", 
    "theverge.com", 
    "technologyreview.com", 
    "arstechnica.com", 
    "cnbc.com", 
    "bloomberg.com", 
    "reuters.com", 
    "marketwatch.com", 
    "investor.com",
    "inc42.com",
    "entrackr.com",
    "yourstory.com",
    "vccircle.com",
    "indianstartupnews.com"
]

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
    """Callback to add a time filter to search queries to get news from the last 6 months."""
    if tool.name == "google_search":
        query = args.get("query", "")
        # tbs=qdr:m6 means "last 6 months"
        if "tbs=qdr:" not in query:
            args['query'] = f"{query} tbs=qdr:m6"
            print(f"MODIFIED query for 6-month freshness: '{args['query']}'")
    return None

def initialize_process_log(tool_context: ToolContext):
    """Helper to ensure the process_log list exists in the state."""
    if 'process_log' not in tool_context.state:
        tool_context.state['process_log'] = []

def inject_process_log_after_search(tool, args, tool_context, tool_response):
    """
    Callback: After a successful search, this injects the process_log and URLs into the response.
    """
    if tool.name == "google_search" and isinstance(tool_response, str):
        # Extract all URLs from the search results
        urls = re.findall(r'https?://[^\s\)]+', tool_response)
        unique_urls = list(dict.fromkeys(urls))  # Preserve order while removing duplicates
        unique_domains = sorted(list(set(urlparse(url).netloc for url in urls)))
        
        if unique_domains:
            sourcing_log = f"Action: Sourced news from the following domains: {', '.join(unique_domains)}."
            current_log = tool_context.state.get('process_log', [])
            tool_context.state['process_log'] = [sourcing_log] + current_log

        # Store URLs in context for later retrieval
        if 'all_urls' not in tool_context.state:
            tool_context.state['all_urls'] = []
        tool_context.state['all_urls'].extend(unique_urls)

        final_log = tool_context.state.get('process_log', [])
        final_urls = tool_context.state.get('all_urls', [])
        
        print(f"CALLBACK LOG: Injecting process log and URLs into tool response")
        print(f"Total URLs found: {len(final_urls)}")
        
        return {
            "search_results": tool_response,
            "process_log": final_log,
            "source_urls": final_urls
        }
    return tool_response



root_agent = Agent(
    name="finance_news_researcher",
    model="gemini-2.5-flash", 
    instruction="""
    **Your Core Identity:**
    You are a Finance News Report Producer. Your job is to orchestrate a complete workflow: find news from the LAST 6 MONTHS on a specific user-provided topic (typically a company), compile a comprehensive report with summaries, and include ALL source URLs.

    **Topic Information:**
    The user has provided a finance topic (usually a company name like "Zoho") for research. Your research scope is strictly limited to this specific topic only. All search queries and analysis must be focused on gathering information relevant to this topic from the LAST 6 MONTHS.

    **Crucial Rules:**
    1.  **6-Month Time Window:** You MUST search for news from the last 6 months only. The callback will enforce this with tbs=qdr:m6.
    2.  **Comprehensive Summary:** Analyze ALL the news articles found and create a comprehensive summary that covers the key themes, developments, and trends over the 6-month period.
    3.  **Include ALL URLs and Dates:** You MUST extract and include ALL source URLs **along with their publication dates** from the search results in both the individual story entries AND in the `all_source_urls` field of the report; if a date is unavailable, use "Not Available".
    4.  **Resilience is Key:** If you encounter an error or cannot find specific information for one item, you MUST NOT halt the entire process. Use placeholder values like "Not Available", and continue to the next step.
    5.  **Topic-Specific Focus:** Your research is strictly limited to the user-specified finance topic.
    6.  **User-Facing Communication:** Your interaction has only two user-facing messages: the initial acknowledgment and the final confirmation. All complex work must happen silently in the background.
    7.  **Save to Output Folder:** Reports must be saved to the 'output' folder using the save_news_to_markdown tool.

    **Understanding Callback-Modified Tool Outputs:**
    The `google_search` tool is enhanced by callbacks. Its final output is a JSON object with three keys:
    1.  `search_results`: A string containing the actual search results.
    2.  `process_log`: A list of strings describing the filtering actions performed.
    3.  `source_urls`: A list of all URLs found in the search results.

    **Required Conversational Workflow:**
    1.  **Acknowledge and Inform:** The VERY FIRST thing you do is respond to the user with: "Okay, I'll start researching news from the last 6 months on your topic. I will compile a comprehensive summary and include all source URLs. This might take a moment."
    2.  **Search (Background Step):** Immediately after acknowledging, use the `google_search` tool to find relevant news from the last 6 months. Your query must be specifically tailored to find news about the user's topic.
    3.  **Analyze & Extract Information (Internal Step):** 
        - Process search results to identify key companies, entities, stories, and their relevant details
        - Extract ALL URLs from the `source_urls` field
        - Create a comprehensive summary of all findings from the 6-month period
        - If financial data cannot be found, use 'N/A'
    4.  **Get Financial Data (Background Step):** Call the `get_financial_context` tool with any extracted tickers. If the tool returns "Not Available" for any ticker, accept this and proceed.
    5.  **Structure the Report (Internal Step):** 
        - Use the `AINewsReport` schema to structure all gathered information
        - Populate `all_source_urls` with ALL URLs from the search results
        - Include URLs in individual story entries
        - If financial data was not found, use "Not Available" in the `financial_context` field
        - Populate the `process_log` field with the log from the search tool
    6.  **Format for Markdown (Internal Step):** 
        - Convert the structured `AINewsReport` data into a well-formatted Markdown string
        - Include a "## Data Sourcing Notes" section with the process_log
        - Include a "## Source URLs" section at the end with ALL URLs as a numbered list
    7.  **Save the Report (Background Step):** Save the Markdown string using `save_news_to_markdown` with the filename `finance_research_report.md`. This will automatically save to the 'output' folder.
    8.  **Final Confirmation:** After successful save, confirm to the user: "Report generated successfully! The report has been saved to output/finance_research_report.md and includes all source URLs."
    """,
    tools=[
        google_search,
        get_financial_context,
        save_news_to_markdown
    ],
    before_tool_callback=[
        filter_news_sources_callback,
        enforce_data_freshness_callback,
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)