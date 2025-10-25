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
    source_name: str = Field(description="Publication name (e.g., Economic Times, YourStory, Inc42)")
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
    instruction=f"""You are an AI News Research Assistant specializing in Indian startup news coverage for the current year.

        Current year: {datetime.now().year}

        For the given Indian startup company, search and extract news articles from {datetime.now().year}:

        SEARCH STRATEGY:
        - Use queries: "[Company] news {datetime.now().year}", "[Company] funding {datetime.now().year}", "[Company] India"
        - Focus on: Economic Times, YourStory, Inc42, Entrackr, The Ken, Mint, VCCircle

        CRITICAL: Use Google Search tool to find articles. For EACH article found, extract:
        1. headline: Clean text without quotes or special characters
        2. summary: 2-3 sentences, avoid quotes and apostrophes (use simple language)
        3. source_url: Full valid URL
        4. source_name: Publication name
        5. published_date: YYYY-MM-DD format if available, otherwise null
        6. category: One of - Funding, Product, Expansion, Partnership, Leadership, Awards, Challenges, Regulatory, Other
        7. key_points: List of 3-5 simple bullet points

        TEXT CLEANING RULES:
        - Replace quotes with simple text
        - Avoid apostrophes - use "is" instead of "'s"
        - Keep text simple and factual
        - Remove any special characters

        EXCLUSION RULES:
        - Skip articles without valid URLs
        - Skip minor mentions or low-quality content
        - Only include substantive news articles

        Find at least 10-15 quality articles if available.

        Use Google Search extensively.""",
    description=f"Extracts news from {datetime.now().year}.",
    tools=[google_search],
    output_key="current_year_news"
)

# Agent 2: Previous Year News
previous_year_agent = LlmAgent(
    name="PreviousYearNewsAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an AI News Research Assistant for Indian startup news from {datetime.now().year - 1}.

        For the given company, search for {datetime.now().year - 1} news:

        SEARCH: "[Company] news {datetime.now().year - 1}", "[Company] {datetime.now().year - 1}"

        SOURCES: YourStory, Inc42, Economic Times, VCCircle, The Ken

        FOR EACH ARTICLE:
        - Clean headline (no quotes or special chars)
        - Simple summary (2-3 sentences, plain text)
        - Valid source_url (MANDATORY)
        - source_name
        - published_date (YYYY-MM-DD or null)
        - category (Funding/Product/Expansion/Partnership/Leadership/Awards/Challenges/Regulatory/Other)
        - key_points (3-5 simple bullets)

        TEXT RULES:
        - Use plain simple English
        - No quotes, apostrophes, or special characters
        - Keep it factual and clean

        Find 10-15 articles if available.

        Use Google Search tool.""",
    description=f"Extracts news from {datetime.now().year - 1}.",
    tools=[google_search],
    output_key="previous_year_news"
)

# Agent 3: Two Years Ago News
two_years_ago_agent = LlmAgent(
    name="TwoYearsAgoNewsAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an AI News Research Assistant for {datetime.now().year - 2}.

        For the given company, find news from {datetime.now().year - 2}:

        SEARCH: "[Company] {datetime.now().year - 2}", "[Company] founded {datetime.now().year - 2}"

        Extract 8-10 articles with:
        - Clean text (no special characters)
        - Valid URLs only
        - Simple summaries
        - Proper categorization

        Use Google Search.""",
    description=f"Extracts news from {datetime.now().year - 2}.",
    tools=[google_search],
    output_key="two_years_ago_news"
)

# Agent 4: Funding-Specific News
funding_news_agent = LlmAgent(
    name="FundingNewsAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for startup funding news.

        SEARCH QUERIES:
        - "[Company] raises funding"
        - "[Company] Series A B C seed"
        - "[Company] investors valuation"

        Focus on funding rounds: Seed, Pre-seed, Series A/B/C/D, bridge rounds

        FOR EACH FUNDING ARTICLE:
        - Clean headline and summary
        - Amount and round type in summary
        - Valid source_url (MANDATORY)
        - Category: Funding
        - Simple text (no quotes or special chars)

        SOURCES: VCCircle, Inc42, YourStory, Entrackr, company press releases

        Find 5-10 funding articles.

        Use Google Search.""",
    description="Extracts funding news.",
    tools=[google_search],
    output_key="funding_news_data"
)

# Agent 5: Product & Technology News
product_news_agent = LlmAgent(
    name="ProductNewsAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for product news.

        SEARCH:
        - "[Company] launches product"
        - "[Company] new feature"
        - "[Company] app platform"

        Extract product launches, updates, tech innovations.

        FOR EACH:
        - Clean headline
        - Simple summary (product name, features)
        - Valid URL
        - Category: Product
        - No special characters

        Find 5-10 articles.

        Use Google Search.""",
    description="Extracts product news.",
    tools=[google_search],
    output_key="product_news_data"
)

# Agent 6: Leadership & Team News
leadership_news_agent = LlmAgent(
    name="LeadershipNewsAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for leadership news.

        SEARCH:
        - "[Company] appoints CEO CTO CFO"
        - "[Company] founder team"
        - "[Company] executive hire"

        Extract C-level appointments, founder changes, key hires.

        FOR EACH:
        - Clean text (person name, role)
        - Simple summary
        - Valid URL
        - Category: Leadership

        Find 5-8 articles.

        Use Google Search.""",
    description="Extracts leadership news.",
    tools=[google_search],
    output_key="leadership_news_data"
)

# Agent 7: Expansion & Partnerships
expansion_news_agent = LlmAgent(
    name="ExpansionNewsAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for expansion news.

        SEARCH:
        - "[Company] expands to"
        - "[Company] partnership"
        - "[Company] new market"

        Extract geographic expansion, partnerships, acquisitions.

        FOR EACH:
        - Clean headline
        - Simple summary
        - Valid URL
        - Category: Expansion or Partnership

        Find 5-8 articles.

        Use Google Search.""",
    description="Extracts expansion news.",
    tools=[google_search],
    output_key="expansion_news_data"
)

# Agent 8: Challenges & Controversies
challenges_news_agent = LlmAgent(
    name="ChallengesNewsAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant for challenges.

        SEARCH:
        - "[Company] layoffs"
        - "[Company] controversy legal"
        - "[Company] challenge issue"

        Extract layoffs, legal issues, controversies (factually, without bias).

        FOR EACH:
        - Clean headline
        - Factual summary
        - Valid URL
        - Category: Challenges

        Find 3-5 articles if available.

        Use Google Search.""",
    description="Extracts challenge news.",
    tools=[google_search],
    output_key="challenges_news_data"
)

# Create ParallelAgent
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
    description="Runs multiple news extraction agents in parallel."
)

# News Synthesis Agent
news_synthesis_agent = LlmAgent(
    name="NewsSynthesisAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an AI News Analyst creating a structured news report in valid JSON format.

        **Input Data from Agents:**
        {{current_year_news}}
        {{previous_year_news}}
        {{two_years_ago_news}}
        {{funding_news_data}}
        {{product_news_data}}
        {{leadership_news_data}}
        {{expansion_news_data}}
        {{challenges_news_data}}

        **CRITICAL: OUTPUT MUST BE VALID JSON**
        
        **BEFORE CREATING JSON - TEXT CLEANING:**
        For ALL text fields (headlines, summaries, descriptions):
        1. Remove ALL quotation marks (both single and double quotes)
        2. Replace possessives: "Company's product" becomes "Company product"
        3. Use only alphanumeric characters, spaces, hyphens, and periods
        4. Keep sentences simple and declarative
        5. No special characters or punctuation inside strings except periods and commas

        **DATA CLEANING:**
        1. EXCLUDE articles without valid source_url
        2. REMOVE duplicate articles (same URL = duplicate)
        3. Sort articles by published_date within each year (newest first)
        4. Ensure dates are "YYYY-MM-DD" format or null
        
        **EXAMPLE OF CLEAN TEXT:**
        BAD: "Sarvam AI's CEO says 'innovation is key' for growth"
        GOOD: "Sarvam AI CEO discusses innovation and company growth strategy"
        
        BAD: The company raised $41M in "series A" funding
        GOOD: The company raised 41 million USD in Series A funding round

        **JSON OUTPUT STRUCTURE:**
        
        Output ONLY this JSON structure with NO additional text, markdown, or explanation:

        {{
          "company_name": "[Clean company name]",
          "analysis_period": "{datetime.now().year - 2}-{datetime.now().year}",
          "total_articles_found": [count all unique articles],
          "news_by_year": [
            {{
              "year": {datetime.now().year},
              "article_count": [count],
              "articles": [
                {{
                  "headline": "[Clean text - no quotes]",
                  "summary": "[Clean text - 2-3 sentences - no quotes]",
                  "source_url": "[full URL]",
                  "source_name": "[publication name]",
                  "published_date": "[YYYY-MM-DD or null]",
                  "category": "[Funding/Product/Expansion/Partnership/Leadership/Awards/Challenges/Regulatory/Other]",
                  "key_points": ["[point 1 - clean]", "[point 2 - clean]", "[point 3 - clean]"]
                }}
              ],
              "year_summary": "[2-3 sentences about this year - clean text]"
            }},
            {{
              "year": {datetime.now().year - 1},
              "article_count": [count],
              "articles": [[article objects with clean text]],
              "year_summary": "[clean text]"
            }},
            {{
              "year": {datetime.now().year - 2},
              "article_count": [count],
              "articles": [[article objects with clean text]],
              "year_summary": "[clean text]"
            }}
          ],
          "funding_news": {{
            "total_funding_articles": [count],
            "funding_timeline": [[funding articles - sorted by date]],
            "total_funding_summary": "[clean text summary]"
          }},
          "product_news": {{
            "total_product_articles": [count],
            "product_milestones": [[product articles]],
            "product_evolution_summary": "[clean text]"
          }},
          "leadership_news": {{
            "total_leadership_articles": [count],
            "leadership_changes": [[leadership articles]],
            "leadership_summary": "[clean text]"
          }},
          "expansion_news": {{
            "total_expansion_articles": [count],
            "expansion_milestones": [[expansion articles]],
            "expansion_summary": "[clean text]"
          }},
          "challenges_news": {{
            "total_challenge_articles": [count],
            "challenges": [[challenge articles]],
            "challenges_summary": "[clean text]"
          }},
          "major_milestones": ["[milestone 1 - clean]", "[milestone 2 - clean]"],
          "overall_trajectory": "[2-3 sentences - clean text]",
          "media_sentiment": "[Positive or Mixed or Negative]",
          "key_themes": ["[theme 1]", "[theme 2]", "[theme 3]"]
        }}

        **CRITICAL JSON RULES:**
        - All string values must NOT contain unescaped quotes
        - Use only plain text in all strings
        - No newlines within string values
        - Arrays must be properly comma-separated
        - No trailing commas in arrays or objects
        - All URLs must be complete and valid
        - Boolean null values for missing dates
        
        **FINAL CHECK BEFORE OUTPUT:**
        - Verify all quotes are properly closed
        - Ensure no apostrophes in possessives
        - Check all arrays are properly formatted
        - Confirm JSON is valid (no syntax errors)

        OUTPUT: Pure JSON only, starting with {{ and ending with }}. No markdown, no backticks, no explanation.""",
    description="Synthesizes news into valid JSON report.",
    output_schema=CompanyNewsProfile,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True
)

# Create Sequential Pipeline
news_analysis_pipeline = SequentialAgent(
    name="NewsAnalysisPipeline",
    sub_agents=[
        parallel_news_extraction,
        news_synthesis_agent
    ],
    description="Coordinates parallel news extraction and synthesizes report."
)

# Root agent
root_agent = news_analysis_pipeline