# Job Fit Analyzer

Paste a job description and get a fit score against your resume. The LangGraph workflow extracts requirements, scores your match, identifies strengths and gaps, and recommends actions to close them. Results are saved to Airtable.

**Live demo:** https://jobfitanalyzer-uao34brzbaiuxiwtydighk.streamlit.app/

## Tech Stack

LangGraph / LangSmith / OpenAI / Airtable / Streamlit

## How it works

1. Extracts key requirements from the job description
2. Scores resume fit against those requirements (0-100)
3. Identifies strengths and gaps
4. Recommends specific actions to close gaps
5. Saves analysis to Airtable

## Run locally

1. Clone the repo
2. Copy `.env.example` to `.env` and add your keys
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
