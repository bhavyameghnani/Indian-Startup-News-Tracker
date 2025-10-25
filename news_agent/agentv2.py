from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import List, Optional

# Model configuration
GEMINI_MODEL = "gemini-2.0-flash"

# --- Enhanced Pydantic Models with Citations ---

class CitedValue(BaseModel):
    """Base model for values that need citations."""
    value: str = Field(description="The actual data value")
    source_url: Optional[str] = Field(description="URL to the source", default=None)
    source_name: Optional[str] = Field(description="Name of the source", default=None)

class CompanyBasicInfo(BaseModel):
    """Basic company information model."""
    company_name: str = Field(description="Official company name")
    logo_url: Optional[str] = Field(description="URL to company logo", default=None)
    headquarters_location: str = Field(description="City, state, and country of headquarters")
    year_founded: Optional[int] = Field(description="Year the company was founded", default=None)
    company_type: str = Field(description="Private, Public, or Subsidiary")
    industry_sector: str = Field(description="Primary industry classification")
    business_model: str = Field(description="B2B, B2C, SaaS, Marketplace, etc.")
    company_stage: CitedValue = Field(description="Company stage with source")
    employee_count: Optional[CitedValue] = Field(description="Employee count with source", default=None)
    website_url: str = Field(description="Official website URL")
    company_description: str = Field(description="Short description or tagline")

class FinancialData(BaseModel):
    """Financial and funding information with citations."""
    total_equity_funding: Optional[CitedValue] = Field(description="Total funding with source", default=None)
    latest_funding_round: Optional[CitedValue] = Field(description="Latest round with source", default=None)
    valuation: Optional[CitedValue] = Field(description="Valuation with source", default=None)
    revenue_growth_rate: Optional[CitedValue] = Field(description="Growth rate with source", default=None)
    financial_strength: str = Field(description="Assessment of financial health")
    key_investors: List[str] = Field(description="List of major investors", default=[])

class KeyPerson(BaseModel):
    """Key person with citation."""
    name: str = Field(description="Person's full name")
    role: str = Field(description="Job title or role")
    background: str = Field(description="Brief professional background")
    source_url: Optional[str] = Field(description="Source URL (LinkedIn, company page)", default=None)

class PeopleData(BaseModel):
    """People and leadership information."""
    key_people: List[KeyPerson] = Field(description="Key leadership with sources", default=[])
    employee_growth_rate: Optional[str] = Field(description="Rate of team growth", default=None)
    hiring_trends: str = Field(description="Current hiring patterns and insights")

class MarketData(BaseModel):
    """Market data with citations for key claims."""
    market_size: Optional[CitedValue] = Field(description="TAM with source", default=None)
    competitive_landscape: CitedValue = Field(description="Competitor analysis with source")
    market_position: str = Field(description="Company's market position")
    competitive_advantages: List[CitedValue] = Field(description="Key differentiators with sources", default=[])
    product_market_fit: str = Field(description="Assessment of product-market fit")

class NewsItem(BaseModel):
    """Individual news item with citation."""
    headline: str = Field(description="News headline or summary")
    source_url: str = Field(description="URL to the news article")
    source_name: str = Field(description="Publication name")
    date: Optional[str] = Field(description="Publication date", default=None)

class ReputationData(BaseModel):
    """Reputation data with proper citations."""
    customer_satisfaction: Optional[CitedValue] = Field(description="Customer satisfaction with source", default=None)
    news_mentions_count: Optional[int] = Field(description="Number of recent mentions", default=None)
    notable_news: List[NewsItem] = Field(description="Key news items with links", default=[])
    partnerships: List[CitedValue] = Field(description="Partnerships with announcement sources", default=[])
    brand_sentiment: str = Field(description="Overall brand sentiment analysis")

class CompanyProfile(BaseModel):
    """Complete company profile model."""
    company_info: CompanyBasicInfo = Field(description="Basic company information")
    financial_data: FinancialData = Field(description="Financial and funding information")
    people_data: PeopleData = Field(description="Leadership and team information")
    market_data: MarketData = Field(description="Market position and competitive data")
    reputation_data: ReputationData = Field(description="Reputation and customer information")
    extraction_summary: str = Field(description="Key insights and overall assessment")

# Enhanced Financial Agent with Citations
financial_agent = LlmAgent(
    name="FinancialAgent", 
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in company financial data extraction with source citations.

        For the given company, research and extract financial information with MANDATORY source citations:

        REQUIRED FORMAT for each financial metric:
        - Value: [the actual data]
        - Source URL: [direct link to the source]
        - Source Name: [name of the publication/site]

        Extract these with citations:
        - Total Equity Funding (cite Crunchbase, PitchBook, or press releases)
        - Latest Funding Round (cite official announcement or funding database)
        - Valuation (cite the specific report or announcement)
        - Revenue Growth Rate (cite financial reports, news articles, or SEC filings)
        - Key Investors list

        CRITICAL: Always include the direct URL and source name for funding amounts, valuations, and growth metrics.
        
        Use Google Search to find Crunchbase, company press releases, SEC filings, and financial news.
        
        Output in structured format with clear value-source pairs for each metric.""",
    description="Extracts financial metrics with mandatory source citations.",
    tools=[google_search],
    output_key="financial_data"
)

# Enhanced People Agent with Citations
people_agent = LlmAgent(
    name="PeopleAgent",
    model=GEMINI_MODEL, 
    instruction="""You are an AI Research Assistant for leadership information with source citations.

        For each key person, provide:
        - Name and Role
        - Background (brief)
        - Source URL (LinkedIn profile, company bio page, or press release)

        Focus on C-level executives and founders only (CEO, CTO, COO, CMO, Founders).
        
        CRITICAL: Each key person MUST include a source URL (preferably LinkedIn or official company page).
        
        Also extract general hiring trends and team growth insights.
        
        Use Google Search for LinkedIn profiles and company About/Team pages.
        
        Output in structured format with source URLs for each person.""",
    description="Extracts leadership profiles with source citations.",
    tools=[google_search],
    output_key="people_data"
)

# Enhanced Market Agent with Citations  
market_agent = LlmAgent(
    name="MarketAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for market analysis with source citations.

        Extract these market metrics with MANDATORY citations:
        - Market Size/TAM (cite specific market research report)
        - Competitive Landscape analysis (cite industry report or analysis)
        - Competitive Advantages (cite sources that mention these advantages)

        REQUIRED FORMAT:
        - Value: [the market data/analysis]
        - Source URL: [direct link to report/article]
        - Source Name: [research firm, publication, or report name]

        Use Google Search for industry reports, market research, and competitive analysis from reputable sources.
        
        CRITICAL: All market claims must include source URLs and names.
        
        Output in structured format with clear citations for each claim.""",
    description="Extracts market data with mandatory source citations.",
    tools=[google_search],
    output_key="market_data"
)

# Enhanced Reputation Agent with News Citations
reputation_agent = LlmAgent(
    name="ReputationAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for reputation and news analysis with citations.

        Extract reputation data with these citation requirements:

        FOR NOTABLE NEWS - MANDATORY format for each news item:
        - Headline: [brief headline or summary]
        - Source URL: [direct link to the article]
        - Source Name: [publication name like TechCrunch, Reuters, etc.]
        - Date: [publication date if available]

        FOR PARTNERSHIPS - Include source URLs to partnership announcements.

        FOR CUSTOMER SATISFACTION - Cite review sites, survey reports, or customer testimonials with URLs.

        Focus on recent news (last 12 months) from credible sources.
        
        CRITICAL: Every news item and partnership MUST include direct URLs to sources.
        
        Use Google Search for recent news, press releases, and customer review sites.
        
        Output in structured format with complete citation information.""",
    description="Extracts reputation data with mandatory news citations.",
    tools=[google_search],
    output_key="reputation_data"
)

# Keep existing company_info_agent unchanged
company_info_agent = LlmAgent(
    name="CompanyInfoAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in company basic information extraction.

        For the given company, research and extract the following basic information:
        - Company Name
        - Headquarters Location (City, State, Country)  
        - Year Founded
        - Company Type (Private/Public/Subsidiary)
        - Industry/Sector classification
        - Business Model (B2B/B2C/SaaS/Marketplace/etc.)
        - Company Stage (with source citation including URL and source name)
        - Employee Headcount (with source citation including URL and source name)
        - Official Website URL
        - Company Description/Tagline

        For Company Stage and Employee Count, provide:
        - Value: [the actual data]
        - Source URL: [direct link]
        - Source Name: [site name like LinkedIn, Crunchbase]

        Use Google Search to find official sources like company website, LinkedIn, Crunchbase.

        Output in structured format with citations for stage and employee count.""",
    description="Extracts basic company information with selective citations.",
    tools=[google_search],
    output_key="company_info_data"
)

# Create the ParallelAgent
parallel_extraction_agent = ParallelAgent(
    name="ParallelCompanyDataExtraction",
    sub_agents=[
        company_info_agent,
        financial_agent,
        people_agent,
        market_agent,
        reputation_agent
    ],
    description="Runs multiple company data extraction agents in parallel with citation requirements."
)

# Enhanced Data Synthesis Agent
data_synthesis_agent = LlmAgent(
    name="DataSynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Assistant responsible for combining company research findings into a comprehensive, structured company profile with proper citations.

        CRITICAL: Clean up the data before final output:
        
        1. For NewsItems: If source_url is empty string ("") or null, REMOVE that news item completely
        2. For CitedValues: If both source_url AND source_name are null/empty, convert to regular string field  
        3. For KeyPerson: If source_url is null/empty, keep the person but set source_url to null
        4. Keep ALL data points - never remove company information, just clean up citation formatting

        **Input Data:**
        {company_info_data}
        {financial_data}
        {people_data}
        {market_data}
        {reputation_data}

        **Data Cleaning Rules:**
        - Remove news items with no source URLs
        - For market data, competitive advantages, partnerships: if no source URL available, present as regular strings without citation structure
        - Keep all financial data with citations (these are critical)
        - Keep all people data but allow null source_urls
        - Maintain all company information

        **Output Format Requirements:**

        # Company Profile: [Company Name]

        ## Executive Summary
        [2-3 sentence overview based on collected data]

        ## Company Overview
        - **Company Name:** 
        - **Founded:** 
        - **Headquarters:** 
        - **Industry:** 
        - **Business Model:** 
        - **Stage:** [Value] ([Source Name](Source URL) if available)
        - **Employees:** [Value] ([Source Name](Source URL) if available)
        - **Website:** 

        ## Financial Profile  
        - **Total Funding:** [Value] ([Source Name](Source URL) if available)
        - **Latest Round:** [Value] ([Source Name](Source URL) if available)
        - **Valuation:** [Value] ([Source Name](Source URL) if available)
        - **Revenue Growth:** [Value] ([Source Name](Source URL) if available)
        - **Key Investors:** [List]

        ## Leadership & Team
        - **Key Leadership:** 
          - [Name], [Role] - [Background] ([Source Name](Source URL) if available)
        - **Team Growth:** 
        - **Hiring Trends:** 

        ## Market Position
        - **Market Size:** [Value] ([Source Name](Source URL) if available)
        - **Competitors:** [Analysis] ([Source Name](Source URL) if available)
        - **Competitive Advantages:** 
          - [Advantage] ([Source Name](Source URL) if available)

        ## Reputation & Customer Voice
        - **Customer Satisfaction:** [Value] ([Source Name](Source URL) if available)
        - **Recent News:** [ONLY include news items that have actual source URLs]
          - [Headline] - [Source Name](Source URL) [Date if available]
        - **Partnerships:** 
          - [Partnership] ([Source Name](Source URL) if available)
        - **Brand Sentiment:** 

        ## Key Insights & Summary
        [Synthesize findings with source-backed claims]

        CRITICAL: 
        - Only include citation links where source URLs are actually provided
        - Remove any news items without source URLs  
        - Convert CitedValue objects to plain strings if they lack proper citations
        - Preserve all actual company data while cleaning citation format""",
    description="Synthesizes company data into structured profile with cleaned citations.",
    output_schema=CompanyProfile,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True
)

# Create the SequentialAgent
company_analysis_pipeline = SequentialAgent(
    name="CompanyAnalysisPipeline",
    sub_agents=[
        parallel_extraction_agent,
        data_synthesis_agent
    ],
    description="Coordinates parallel data extraction with citations and synthesizes comprehensive company profile."
)

# Main agent
root_agent = company_analysis_pipeline