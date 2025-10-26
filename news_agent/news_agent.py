from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Model configuration
GEMINI_MODEL = "gemini-2.0-flash"

# --- Pydantic Models for News Aggregation ---

class NewsArticle(BaseModel):
    """Individual news article with complete citation."""
    headline: str = Field(description="Article headline or title")
    summary: str = Field(description="2-3 sentence summary of the article content")
    source_url: str = Field(description="Direct URL to the article")
    source_name: str = Field(description="Publication name")
    published_date: Optional[str] = Field(description="Publication date (YYYY-MM-DD format)", default=None)
    category: str = Field(description="News category: Funding, Product, Expansion, Partnership, Leadership, Awards, Challenges, Regulatory, Other")
    key_points: List[str] = Field(description="3-5 key takeaways from the article", default=[])

class YearlyNews(BaseModel):
    """News articles grouped by year."""
    year: int = Field(description="Calendar year")
    article_count: int = Field(description="Total number of articles for this year")
    articles: List[NewsArticle] = Field(description="List of news articles", default=[])
    year_summary: str = Field(description="Executive summary of major events this year")

class FundingNews(BaseModel):
    """Dedicated section for funding-related news."""
    total_funding_articles: int = Field(description="Count of funding articles")
    funding_timeline: List[NewsArticle] = Field(description="Funding announcements chronologically", default=[])
    total_funding_summary: str = Field(description="Summary of total funding raised based on news")

class ProductNews(BaseModel):
    """Product launches and updates."""
    total_product_articles: int = Field(description="Count of product-related articles")
    product_milestones: List[NewsArticle] = Field(description="Product launches, updates, and milestones", default=[])
    product_evolution_summary: str = Field(description="Summary of product development trajectory")

class LeadershipNews(BaseModel):
    """Leadership changes and appointments."""
    total_leadership_articles: int = Field(description="Count of leadership articles")
    leadership_changes: List[NewsArticle] = Field(description="Leadership appointments, exits, and changes", default=[])
    leadership_summary: str = Field(description="Summary of leadership evolution")

class ExpansionNews(BaseModel):
    """Business expansion and growth news."""
    total_expansion_articles: int = Field(description="Count of expansion articles")
    expansion_milestones: List[NewsArticle] = Field(description="Market expansion, new offices, international growth", default=[])
    expansion_summary: str = Field(description="Summary of geographic and business expansion")

class ChallengesNews(BaseModel):
    """Challenges, controversies, and negative news."""
    total_challenge_articles: int = Field(description="Count of challenge/controversy articles")
    challenges: List[NewsArticle] = Field(description="Challenges, layoffs, controversies, legal issues", default=[])
    challenges_summary: str = Field(description="Summary of major challenges faced")

class CompanyNewsProfile(BaseModel):
    """Complete news profile for a startup company."""
    company_name: str = Field(description="Official company name")
    analysis_period: str = Field(description="Time period covered (e.g., '2022-2025')")
    total_articles_found: int = Field(description="Total number of articles found")
    
    # Yearly breakdown
    news_by_year: List[YearlyNews] = Field(description="News organized by year", default=[])
    
    # Category-based organization
    funding_news: FundingNews = Field(description="All funding-related news")
    product_news: ProductNews = Field(description="Product and technology news")
    leadership_news: LeadershipNews = Field(description="Leadership and management news")
    expansion_news: ExpansionNews = Field(description="Business expansion news")
    challenges_news: ChallengesNews = Field(description="Challenges and controversies")
    
    # Overall insights
    major_milestones: List[str] = Field(description="Top 5-10 major milestones", default=[])
    overall_trajectory: str = Field(description="Overall company trajectory analysis")
    media_sentiment: str = Field(description="Overall media sentiment: Positive, Mixed, or Negative")
    key_themes: List[str] = Field(description="Recurring themes in news coverage", default=[])

# --- Specialized News Agents ---

# Agent 1: Recent News (Current Year)
current_year_agent = LlmAgent(
    name="CurrentYearNewsAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an expert AI News Research Assistant specializing in startup coverage.

Current year: {datetime.now().year}

For the given company, use Google Search to find news articles from {datetime.now().year}.

SEARCH STRATEGY:
- "[Company] news {datetime.now().year}"
- "[Company] funding {datetime.now().year}"
- "[Company] startup"

Search ALL credible news sources - no restrictions on domains.

For EACH article found, extract:
- headline: Clear article title
- summary: 2-3 sentences describing the news
- source_url: Complete valid URL (MANDATORY)
- source_name: Publication name
- published_date: YYYY-MM-DD format if available
- category: Funding/Product/Expansion/Partnership/Leadership/Awards/Challenges/Regulatory/Other
- key_points: 3-5 bullet points

EXCLUSIONS:
- Skip articles without URLs
- Skip minor mentions
- Only substantive news

Find 10-15 quality articles from any reputable source. Use google_search tool extensively.""",
    description=f"Extracts {datetime.now().year} news",
    tools=[google_search],
    output_key="current_year_news"
)

# Agent 2: Previous Year
previous_year_agent = LlmAgent(
    name="PreviousYearNewsAgent",
    model=GEMINI_MODEL,
    instruction=f"""Extract news from {datetime.now().year - 1} for the given company.

SEARCH: "[Company] news {datetime.now().year - 1}", "[Company] {datetime.now().year - 1}"

Search all credible news sources without domain restrictions.

Extract same fields as current year agent. Find 10-15 articles.

Use google_search tool.""",
    description=f"Extracts {datetime.now().year - 1} news",
    tools=[google_search],
    output_key="previous_year_news"
)

# Agent 3: Two Years Ago
two_years_ago_agent = LlmAgent(
    name="TwoYearsAgoNewsAgent",
    model=GEMINI_MODEL,
    instruction=f"""Extract news from {datetime.now().year - 2} for the given company.

SEARCH: "[Company] {datetime.now().year - 2}"

Search all reputable sources. Extract 8-10 articles with complete information. Use google_search.""",
    description=f"Extracts {datetime.now().year - 2} news",
    tools=[google_search],
    output_key="two_years_ago_news"
)

# Agent 4: Funding News
funding_news_agent = LlmAgent(
    name="FundingNewsAgent",
    model=GEMINI_MODEL,
    instruction="""Extract all funding-related news for the company.

SEARCH:
- "[Company] raises funding"
- "[Company] Series A B C seed"
- "[Company] investors"

Focus on: Seed, Pre-seed, Series A/B/C/D rounds

Category: Funding for all articles

Search all sources. Find 5-10 funding articles. Use google_search.""",
    description="Extracts funding news",
    tools=[google_search],
    output_key="funding_news_data"
)

# Agent 5: Product News
product_news_agent = LlmAgent(
    name="ProductNewsAgent",
    model=GEMINI_MODEL,
    instruction="""Extract product and technology news.

SEARCH:
- "[Company] launches product"
- "[Company] new feature"
- "[Company] platform app"

Category: Product

Find 5-10 articles from any credible source. Use google_search.""",
    description="Extracts product news",
    tools=[google_search],
    output_key="product_news_data"
)

# Agent 6: Leadership News
leadership_news_agent = LlmAgent(
    name="LeadershipNewsAgent",
    model=GEMINI_MODEL,
    instruction="""Extract leadership and team changes.

SEARCH:
- "[Company] appoints CEO CTO CFO"
- "[Company] founder"
- "[Company] executive hire"

Category: Leadership

Find 5-8 articles from any source. Use google_search.""",
    description="Extracts leadership news",
    tools=[google_search],
    output_key="leadership_news_data"
)

# Agent 7: Expansion News
expansion_news_agent = LlmAgent(
    name="ExpansionNewsAgent",
    model=GEMINI_MODEL,
    instruction="""Extract expansion and partnership news.

SEARCH:
- "[Company] expands"
- "[Company] partnership"
- "[Company] new market"

Category: Expansion or Partnership

Find 5-8 articles from any credible source. Use google_search.""",
    description="Extracts expansion news",
    tools=[google_search],
    output_key="expansion_news_data"
)

# Agent 8: Challenges
challenges_news_agent = LlmAgent(
    name="ChallengesNewsAgent",
    model=GEMINI_MODEL,
    instruction="""Extract challenge and controversy news.

SEARCH:
- "[Company] layoffs"
- "[Company] controversy"
- "[Company] legal issue"

Category: Challenges

Find 3-5 articles if available from any credible source. Be factual. Use google_search.""",
    description="Extracts challenges news",
    tools=[google_search],
    output_key="challenges_news_data"
)

# Parallel execution
parallel_news_extraction = ParallelAgent(
    name="ParallelNewsExtraction",
    sub_agents=[
        current_year_agent,
        previous_year_agent,
        two_years_ago_agent,
        funding_news_agent,
        product_news_agent,
        leadership_news_agent,
        expansion_news_agent,
        challenges_news_agent
    ],
    description="Runs all news extraction agents in parallel"
)

# Synthesis Agent with structured output
news_synthesis_agent = LlmAgent(
    name="NewsSynthesisAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an AI News Analyst creating a comprehensive news report.

**Input Data:**
You will receive outputs from multiple news extraction agents containing articles about the company.

**Your Task:**
Create a complete CompanyNewsProfile by:

1. **DATA CLEANING:**
   - Remove duplicate articles (same URL = duplicate)
   - Only include articles with valid source_url
   - Sort articles by date (newest first) within each year

2. **ORGANIZE BY YEAR:**
   - Group articles into years: {datetime.now().year}, {datetime.now().year - 1}, {datetime.now().year - 2}
   - Count articles per year
   - Write a 2-3 sentence summary for each year

3. **ORGANIZE BY CATEGORY:**
   - Funding: All funding rounds and investments
   - Product: Product launches, features, tech updates
   - Leadership: Appointments, founder news, team changes
   - Expansion: Geographic expansion, partnerships, acquisitions
   - Challenges: Layoffs, controversies, legal issues

4. **PROVIDE INSIGHTS:**
   - List 5-10 major milestones
   - Write overall trajectory (2-3 sentences)
   - Determine sentiment: Positive, Mixed, or Negative
   - Identify 3-5 key themes

5. **COUNTS:**
   - total_articles_found: Count ALL unique articles
   - Set article counts for each category and year

**CRITICAL JSON FORMATTING RULES:**
- All string values MUST have quotes and newlines removed
- Replace any quotes within strings with single quotes or remove them
- Keep summaries concise to avoid string issues
- Do not include any line breaks within string values
- Escape any special characters properly

**OUTPUT:**
Return a complete CompanyNewsProfile object with all fields populated.

Be thorough and analytical.""",
    description="Synthesizes all news into structured report",
    output_schema=CompanyNewsProfile,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True
)

# Sequential pipeline
news_analysis_pipeline = SequentialAgent(
    name="NewsAnalysisPipeline",
    sub_agents=[
        parallel_news_extraction,
        news_synthesis_agent
    ],
    description="Extracts news in parallel then synthesizes into report"
)

# Root agent
root_agent = news_analysis_pipeline