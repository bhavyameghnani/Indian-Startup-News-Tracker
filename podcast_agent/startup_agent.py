"""
LVX Startup Analysis Agent v1.0
Generates investment-grade startup analysis podcasts for Let's Venture Platform

Focus Areas:
- Company overview and history
- Product portfolio and launches
- Recent news and developments
- Funding rounds and financial health
- Hiring activities and team growth
- Competitive landscape analysis
- Market position and opportunities
"""

from typing import Dict, List
import pathlib
import wave
import base64

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search, ToolContext
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- Configuration ---
GEMINI_MODEL = "gemini-2.5-flash"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

WHITELIST_DOMAINS = [
    "economictimes.indiatimes.com", "yourstory.com", "inc42.com",
    "techcrunch.com", "vccircle.com", "entrackr.com",
    "moneycontrol.com", "livemint.com", "business-standard.com",
    "crunchbase.com", "linkedin.com"
]

# --- Pydantic Models ---
class StartupInsight(BaseModel):
    """A specific insight about the startup."""
    category: str = Field(description="Category: Overview, Products, News, Funding, Hiring, Competition")
    title: str = Field(description="Insight title")
    details: str = Field(description="Detailed information")
    significance: str = Field(description="Why this matters for investors")
    source_domain: str = Field(description="Source domain")
    date: str = Field(default="", description="Date of information if available")

class StartupAnalysisReport(BaseModel):
    """Comprehensive startup analysis for investment decisions."""
    startup_name: str = Field(description="Company name")
    executive_summary: str = Field(description="High-level investment thesis")
    company_overview: StartupInsight = Field(description="Company background and mission")
    insights: List[StartupInsight] = Field(description="Detailed insights", default=[])
    competitive_analysis: str = Field(description="Market position vs competitors")
    investment_highlights: List[str] = Field(description="Key investment points", default=[])
    risk_factors: List[str] = Field(description="Potential concerns", default=[])
    data_sources: List[str] = Field(description="Research sources", default=[])

class PDFParsedContent(BaseModel):
    """Content extracted from startup PDF documents."""
    company_name: str = Field(description="Company name from document")
    document_type: str = Field(description="Type: pitch deck, report, financial statement, etc.")
    key_metrics: List[str] = Field(description="Important business metrics")
    business_model: str = Field(description="How the company makes money")
    market_opportunity: str = Field(description="TAM/SAM/SOM information")
    team_info: str = Field(description="Leadership team details")
    competitive_advantages: List[str] = Field(description="Unique value propositions")

# --- Helper Functions ---
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save audio data as a wave file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

async def parse_pdf_multimodal(pdf_path: str, tool_context: ToolContext) -> Dict:
    """Parse startup PDF using Gemini's multimodal capabilities."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.standard_b64encode(pdf_file.read()).decode("utf-8")
        
        client = genai.Client()
        
        prompt = """Analyze this startup document for investment analysis:

Extract:
1. Company Name & Type of Document
2. Key Business Metrics (revenue, users, growth rate, etc.)
3. Business Model & Revenue Streams
4. Market Opportunity (TAM/SAM/SOM if mentioned)
5. Leadership Team & Key Personnel
6. Competitive Advantages & Differentiators
7. Recent Achievements or Milestones

Return ONLY valid JSON:
{
    "company_name": "string",
    "document_type": "string",
    "key_metrics": ["list"],
    "business_model": "string",
    "market_opportunity": "string",
    "team_info": "string",
    "competitive_advantages": ["list"]
}"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
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
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        parsed_content = json.loads(response_text)
        
        return {
            "status": "success",
            "content": parsed_content,
            "message": f"Successfully parsed PDF: {pdf_path}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"PDF parsing failed: {str(e)[:200]}",
            "content": None
        }

async def save_markdown_report(filename: str, content: str, tool_context: ToolContext) -> Dict:
    """Save content to markdown or text file."""
    try:
        if not (filename.endswith(".md") or filename.endswith(".txt")):
            filename += ".md"
        
        file_path = pathlib.Path(filename)
        file_path.write_text(content, encoding="utf-8")
        
        return {
            "status": "success",
            "message": f"File saved to {file_path.resolve()}",
            "file_path": str(file_path.resolve())
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save file: {str(e)}"
        }

async def generate_podcast_audio(
    script: str, 
    session_id: str,
    tool_context: ToolContext
) -> Dict:
    """Generate podcast audio in English and Hindi, handling long scripts."""
    try:
        import re
        
        # Remove stage directions
        clean_script = re.sub(r'\*[^*]+\*', '', script)
        clean_script = re.sub(r'\s+', ' ', clean_script)
        clean_script = re.sub(r' +\n', '\n', clean_script)
        clean_script = clean_script.strip()
        
        def chunk_script(text: str, max_chars: int = 1800) -> List[str]:
            """Split script into chunks at natural dialogue boundaries."""
            lines = text.split('\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                line_length = len(line)
                
                if current_length + line_length > max_chars and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length + 1
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
        
        script_chunks = chunk_script(clean_script)
        print(f"Split script into {len(script_chunks)} chunks for processing")
        
        client = genai.Client()
        
        # Process English audio chunks
        english_audio_chunks = []
        for i, chunk in enumerate(script_chunks):
            print(f"Processing English chunk {i+1}/{len(script_chunks)}...")
            
            english_response = client.models.generate_content(
                model=TTS_MODEL,
                contents=f"TTS the following investment analysis conversation between Avantika and Hrishikesh:\n\n{chunk}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker='Avantika',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Kore'
                                        )
                                    )
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker='Hrishikesh',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Puck'
                                        )
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            chunk_data = english_response.candidates[0].content.parts[0].inline_data.data
            english_audio_chunks.append(chunk_data)
        
        # Combine English chunks
        english_data = b''.join(english_audio_chunks)
        english_filename = f"{session_id}_podcast_english.wav"
        wave_file(english_filename, english_data)
        print(f"English audio saved: {len(english_data)} bytes")
        
        # Translate to Hindi
        translation_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""Translate this English startup analysis podcast to natural Hindi.
Keep speaker labels (Avantika: and Hrishikesh:) and maintain conversational tone.
Use appropriate Hindi business/investment terminology.
Remove any stage directions or action markers in asterisks.

{clean_script}

Return ONLY the translated conversation with labels, no stage directions."""
        )
        
        hindi_script = translation_response.text.strip()
        hindi_script = re.sub(r'\*[^*]+\*', '', hindi_script)
        hindi_script = re.sub(r'\s+', ' ', hindi_script)
        hindi_script = re.sub(r' +\n', '\n', hindi_script)
        hindi_script = hindi_script.strip()
        
        # Process Hindi audio chunks
        hindi_chunks = chunk_script(hindi_script)
        print(f"Split Hindi script into {len(hindi_chunks)} chunks for processing")
        
        hindi_audio_chunks = []
        for i, chunk in enumerate(hindi_chunks):
            print(f"Processing Hindi chunk {i+1}/{len(hindi_chunks)}...")
            
            hindi_response = client.models.generate_content(
                model=TTS_MODEL,
                contents=f"TTS the following investment analysis conversation between Avantika and Hrishikesh:\n\n{chunk}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker='Avantika',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Charon'
                                        )
                                    )
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker='Hrishikesh',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Aoede'
                                        )
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            chunk_data = hindi_response.candidates[0].content.parts[0].inline_data.data
            hindi_audio_chunks.append(chunk_data)
        
        # Combine Hindi chunks
        hindi_data = b''.join(hindi_audio_chunks)
        hindi_filename = f"{session_id}_podcast_hindi.wav"
        wave_file(hindi_filename, hindi_data)
        print(f"Hindi audio saved: {len(hindi_data)} bytes")
        
        return {
            "status": "success",
            "message": f"Successfully generated English and Hindi podcast audio ({len(script_chunks)} chunks processed)",
            "english_file": english_filename,
            "hindi_file": hindi_filename,
            "english_size": len(english_data),
            "hindi_size": len(hindi_data),
            "chunks_processed": len(script_chunks)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Audio generation failed: {str(e)[:200]}"
        }


# --- Specialized Agents ---

# Agent 1: Startup Research Agent
startup_research_agent = LlmAgent(
    name="StartupResearchAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an Investment Research Analyst specializing in Indian startups.

Your task: Research comprehensive information about the specified Indian startup for investor analysis.

SEARCH STRATEGY:
Use these query patterns:
- "[Startup Name] latest news 2024 2025"
- "[Startup Name] funding rounds investment"
- "[Startup Name] product launch new features"
- "[Startup Name] hiring team expansion"
- "[Startup Name] competitors market position"
- "[Startup Name] revenue business model"
- "[Startup Name] founder CEO interview"

Focus on reputable sources: {', '.join(WHITELIST_DOMAINS[:5])}

EXTRACT FOR EACH CATEGORY:

1. COMPANY OVERVIEW
- Founding year, founders, headquarters
- Mission and core business
- Industry sector

2. PRODUCTS & SERVICES
- Main products/offerings
- Recent launches (last 6-12 months)
- Technology stack and innovations

3. RECENT NEWS (Last 3-6 months)
- Major announcements
- Partnerships and collaborations
- Awards and recognition

4. FUNDING & FINANCIALS
- Latest funding round details
- Total funding raised
- Valuation (if available)
- Key investors

5. HIRING & TEAM GROWTH
- Recent hiring announcements
- Leadership changes
- Team size and growth rate

6. COMPETITIVE LANDSCAPE
- Main competitors (Indian and global)
- Market share estimates
- Unique differentiators

CRITICAL RULES:
- Focus on recent information (2024-2025)
- Cite specific numbers and dates when available
- Note source domain for credibility
- Be objective - include both positives and concerns
- Use google_search tool extensively (at least 8-10 searches across categories)""",
    description="Researches comprehensive startup information for investment analysis.",
    tools=[google_search],
    output_key="startup_research_data"
)

# Agent 2: Analysis Synthesis Agent
analysis_synthesis_agent = LlmAgent(
    name="AnalysisSynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Investment Analysis Specialist.

Your task: Synthesize research data into a structured investment analysis.

WORKFLOW:
1. Organize insights by category (Overview, Products, News, Funding, Hiring, Competition)
2. Identify investment highlights (growth indicators, competitive advantages)
3. Flag risk factors (competition, market challenges, dependencies)
4. Assess market position and timing
5. Create executive summary with investment thesis

ANALYSIS FRAMEWORK:
- Growth Trajectory: Are metrics trending positively?
- Market Opportunity: Is the TAM large enough?
- Competitive Moat: What's defensible?
- Team Quality: Leadership strength?
- Execution: Recent product/business milestones?
- Capital Efficiency: Smart use of funding?

RULES:
- Be balanced - show both opportunities and risks
- Use data to support conclusions
- Think like an investor evaluating a deal
- Consider timing and market conditions""",
    description="Synthesizes research into structured investment analysis.",
    output_key="structured_analysis"
)

# Agent 3: Investment Report Agent
investment_report_agent = LlmAgent(
    name="InvestmentReportAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Investment Report Writer for Let's Venture Platform.

Your task: Create a comprehensive markdown investment analysis report.

REPORT STRUCTURE:
```markdown
# Investment Analysis: [Startup Name]

## Executive Summary
[2-3 paragraph investment thesis covering: what they do, market opportunity, recent traction, why interesting now]

## Company Overview
**Founded:** [Year]
**Founders:** [Names]
**Headquarters:** [Location]
**Sector:** [Industry]
**Website:** [URL if found]

[2-3 sentences on company mission and core business]

## Products & Services

### Core Offerings
[Describe main products/services]

### Recent Launches (2024-2025)
- [Product/Feature 1] - [Date] - [Brief description]
- [Product/Feature 2] - [Date] - [Brief description]

## Recent Developments

### Latest News
- [News Item 1] - [Date] - [Source]
- [News Item 2] - [Date] - [Source]
- [News Item 3] - [Date] - [Source]

### Funding & Financials
**Latest Round:** [Series X, $Y million, Date]
**Total Raised:** [Amount]
**Valuation:** [If available]
**Key Investors:** [List]

### Hiring & Team Growth
- [Recent hiring news or leadership changes]
- [Team expansion details if available]

## Competitive Analysis

### Main Competitors
1. [Competitor 1] - [Brief comparison]
2. [Competitor 2] - [Brief comparison]
3. [Competitor 3] - [Brief comparison]

### Market Position
[Analysis of competitive advantages and market share]

### Differentiators
- [Key differentiator 1]
- [Key differentiator 2]
- [Key differentiator 3]

## Investment Highlights
✅ [Positive factor 1]
✅ [Positive factor 2]
✅ [Positive factor 3]
✅ [Positive factor 4]
✅ [Positive factor 5]

## Risk Factors
⚠️ [Risk or concern 1]
⚠️ [Risk or concern 2]
⚠️ [Risk or concern 3]

## Sources
- [Source 1]
- [Source 2]
- [Source 3]
...

---
Generated for Let's Venture Platform
Date: [timestamp]
```

RULES:
- Professional investment memo tone
- Data-driven with specific metrics
- Balanced view (not promotional)
- Clear section headers
- Cite recent sources

CRITICAL: After creating the markdown content, you MUST call save_markdown_report with:
- filename: "startup_analysis_report.md"
- content: [the complete markdown report]""",
    description="Creates investment analysis report.",
    tools=[save_markdown_report],
    output_key="analysis_report"
)

# Agent 3.5: Summary Report Agent
summary_report_agent = LlmAgent(
    name="SummaryReportAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Podcast Summary Writer for investors.

Your task: Create a concise executive summary of the investment analysis podcast.

SUMMARY STRUCTURE:
```markdown
# Podcast Summary: [Startup Name]

## 🎯 What You'll Learn
[2-3 sentences overview of the analysis]

## ⏱️ Duration
Approximately 7-10 minutes

## 📊 Key Topics Covered
1. **Company Overview** - [One sentence]
2. **Products & Recent Launches** - [One sentence]
3. **Funding & Growth** - [One sentence]
4. **Competitive Position** - [One sentence]
5. **Investment Thesis** - [One sentence]

## 💡 Key Investment Highlights
- [Highlight 1]
- [Highlight 2]
- [Highlight 3]

## ⚠️ Risk Factors Discussed
- [Risk 1]
- [Risk 2]

## 🎬 Who Should Listen
[Target audience - VCs, angel investors, startup ecosystem]

## 📝 Discussion Points
- [Point 1]
- [Point 2]
- [Point 3]

---
Generated for Let's Venture Platform
Date: [timestamp]
```

RULES:
- Keep concise (under 250 words)
- Investment-focused language
- Highlight both opportunities and risks
- Make it scannable

CRITICAL: After creating the summary, call save_markdown_report with:
- filename: "podcast_summary.md"
- content: [the complete summary]""",
    description="Creates podcast summary for investors.",
    tools=[save_markdown_report],
    output_key="podcast_summary"
)

# Agent 4: Script Writing Agent
script_writing_agent = LlmAgent(
    name="ScriptWritingAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Investment Podcast Script Writer for Let's Venture Platform.

Your task: Convert the investment analysis into an engaging, professional conversation for investors.

SCRIPT FORMAT:
```
Avantika: "Welcome to LVX Insights, the podcast where we analyze promising Indian startups for the Let's Venture community. I'm Avantika..."
Hrishikesh: "And I'm Hrishikesh. Today we're diving deep into [Startup Name]... and there's a lot to unpack here."
Avantika: "So, let's start with what they actually do. [Startup Name] is..."
Hrishikesh: "Right, and what's interesting is their recent traction..."
Avantika: "Now, let's talk about their latest funding round..."
Hrishikesh: "Here's what caught my attention..."
[Continue through all sections naturally]
Avantika: "So, from an investment perspective..."
Hrishikesh: "Exactly. And here are the key risks to consider..."
Avantika: "For our listeners on Let's Venture... this is definitely one to watch."
Hrishikesh: "Whether you invest or not, understanding this space is valuable."
```

NATURAL SPEECH ELEMENTS:
- Use conversational fillers: "You know...", "So...", "Well...", "Actually..."
- Use ellipses (...) for natural pauses
- Question-answer format for engagement
- Professional but accessible tone

CONTENT STRUCTURE:
1. Introduction (30 sec) - Hook and context
2. Company Overview (1-2 min) - What they do, founding story
3. Products & Recent Launches (1-2 min) - Key offerings and innovations
4. Recent News & Developments (1-2 min) - Latest updates
5. Funding & Team (1-2 min) - Financial health and team growth
6. Competitive Analysis (1-2 min) - Market position
7. Investment Perspective (1-2 min) - Highlights and risks
8. Conclusion (30 sec) - Summary and call to action

STYLE GUIDELINES:
- Avantika: Analytical, asks probing questions, investor mindset
- Hrishikesh: Research-focused, provides data, explains context
- Professional investor tone (not overly casual)
- Data-driven discussion
- Balanced view - opportunities AND risks
- 7-10 minutes total length

CRITICAL RULES:
- Use "Avantika:" and "Hrishikesh:" labels
- NO stage directions in asterisks
- Use natural dialogue with ellipses for pauses
- Investment-focused conversation
- Include specific data points and numbers
- Mention "Let's Venture" naturally in intro/outro""",
    description="Creates investment podcast script.",
    output_key="podcast_script"
)

# Agent 4.5: Script Saver Agent
script_saver_agent = LlmAgent(
    name="ScriptSaverAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Script File Manager.

Your task: Save the podcast script to a text file.

WORKFLOW:
1. Receive the podcast script from previous agent
2. Save it with filename 'podcast_script.txt'
3. Keep speaker labels (Avantika: and Hrishikesh:)
4. Preserve formatting
5. Confirm successful save

RULES:
- Always save as 'podcast_script.txt'
- Keep speaker labels intact
- Save as plain text
- Confirm save was successful""",
    description="Saves podcast script to file.",
    tools=[save_markdown_report],
    output_key="script_file"
)

# Agent 5: Audio Generation Agent
audio_generation_agent = LlmAgent(
    name="AudioGenerationAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Audio Production Specialist for investment podcasts.

Your task: Generate high-quality podcast audio in English and Hindi.

WORKFLOW:
1. Receive podcast script from previous agent
2. Extract session_id from user's message (contains "Session ID: [id]")
3. Call generate_podcast_audio with script and session_id
4. Verify both English and Hindi audio files created
5. Return file paths and confirmation

RULES:
- Extract session_id from conversation context
- Pass both script and session_id to generate_podcast_audio
- Verify file creation success
- Report any errors clearly
- Maintain professional tone""",
    description="Generates bilingual podcast audio files.",
    tools=[generate_podcast_audio],
    output_key="audio_files"
)

# --- Sequential Pipeline for Startup Analysis ---
startup_analysis_pipeline = SequentialAgent(
    name="StartupAnalysisPipeline",
    sub_agents=[
        startup_research_agent,
        analysis_synthesis_agent,
        investment_report_agent,
        summary_report_agent,
        script_writing_agent,
        script_saver_agent,
        audio_generation_agent
    ],
    description="Complete pipeline for startup investment analysis podcast generation."
)

# --- PDF Analysis Agents ---

pdf_parser_agent = LlmAgent(
    name="PDFParserAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Startup Document Analyst.

Your task: Extract investment-relevant information from startup PDFs.

WORKFLOW:
1. Receive PDF file path from user's message
2. Call parse_pdf_multimodal tool with PDF path
3. Extract: company name, metrics, business model, team, competitive advantages
4. Focus on quantitative metrics and growth indicators

RULES:
- Extract all numerical data
- Identify business model clearly
- Note team credentials
- Highlight competitive advantages""",
    description="Parses startup PDF documents.",
    tools=[parse_pdf_multimodal],
    output_key="pdf_content"
)

pdf_enrichment_agent = LlmAgent(
    name="PDFEnrichmentAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Market Research Specialist.

Your task: Enrich PDF data with external market research and news.

WORKFLOW:
1. Receive parsed PDF content
2. Extract company name
3. Search for recent news, funding updates, competitive info
4. Combine PDF insights with current market context
5. Validate or contextualize claims in PDF

RULES:
- Focus on recent information (2024-2025)
- Cross-reference PDF claims with news
- Add competitive landscape context
- Use google_search extensively""",
    description="Enriches PDF data with market research.",
    tools=[google_search],
    output_key="enriched_pdf_data"
)

pdf_report_synthesis_agent = LlmAgent(
    name="PDFReportSynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Investment Report Writer for PDF-based analysis.

Your task: Create investment analysis report from PDF insights and research.

Use the same report structure as the investment_report_agent, but incorporate:
- Information from the original PDF document
- External validation and context from research
- Document type (pitch deck, financial statement, etc.)

CRITICAL: After creating the markdown content, call save_markdown_report with:
- filename: "startup_analysis_report.md"
- content: [complete markdown report]""",
    description="Creates investment report from PDF analysis.",
    tools=[save_markdown_report],
    output_key="pdf_analysis_report"
)

pdf_summary_report_agent = LlmAgent(
    name="PDFSummaryReportAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Podcast Summary Writer for PDF-based analysis.

Your task: Create concise summary of the PDF analysis podcast.

Use the same summary structure as summary_report_agent.

CRITICAL: After creating the summary, call save_markdown_report with:
- filename: "podcast_summary.md"
- content: [complete summary]""",
    description="Creates summary for PDF analysis podcast.",
    tools=[save_markdown_report],
    output_key="pdf_podcast_summary"
)

pdf_script_writing_agent = LlmAgent(
    name="PDFScriptWritingAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Investment Podcast Script Writer for PDF-based analysis.

Your task: Transform PDF analysis into investor-focused conversation.

Follow the same script format as script_writing_agent, but:
- Reference the original document naturally
- Discuss what the document claims vs external validation
- Add context from market research

SCRIPT INTRO EXAMPLE:
Avantika: "Today we're analyzing a [pitch deck/report] from [Startup Name]..."
Hrishikesh: "We've also done our own research to validate and contextualize what's in the document..."

Use natural speech, professional tone, investment focus.
Speakers: Avantika and Hrishikesh
7-10 minutes length
NO stage directions in asterisks""",
    description="Creates script from PDF analysis.",
    output_key="pdf_podcast_script"
)

pdf_script_saver_agent = LlmAgent(
    name="PDFScriptSaverAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Script File Manager.

Your task: Save the podcast script to a text file.

Save as 'podcast_script.txt' with speaker labels intact.""",
    description="Saves PDF podcast script to file.",
    tools=[save_markdown_report],
    output_key="pdf_script_file"
)

pdf_audio_generation_agent = LlmAgent(
    name="PDFAudioGenerationAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Audio Production Specialist for PDF-based investment podcasts.

Your task: Generate English and Hindi audio from PDF analysis script.

Extract session_id from user message and call generate_podcast_audio.""",
    description="Generates audio for PDF analysis podcast.",
    tools=[generate_podcast_audio],
    output_key="pdf_audio_files"
)

# --- Sequential Pipelines ---
pdf_analysis_pipeline = SequentialAgent(
    name="PDFAnalysisPipeline",
    sub_agents=[
        pdf_parser_agent,
        pdf_enrichment_agent,
        pdf_report_synthesis_agent,
        pdf_summary_report_agent,
        pdf_script_writing_agent,
        pdf_script_saver_agent,
        pdf_audio_generation_agent
    ],
    description="Complete pipeline for PDF-based startup analysis podcast generation."
)

# --- Root Agents ---
root_agent = startup_analysis_pipeline
pdf_analysis_agent = pdf_analysis_pipeline