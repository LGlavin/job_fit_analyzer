import streamlit as st
import json
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List

load_dotenv()

# ── RESUME ────────────────────────────────────────────────────────────────────
MY_RESUME = """
Elizabeth Glavin
Technical Lead & Senior Technical Support Engineer — Twilio (2021-Present)
- Lead escalation workflows for complex Messaging API issues
- Collaborated with AI engineering on LLM-powered ticket routing
- Built Jira automations reducing manual triage by 30%
- Mentored global support engineers
- Tools: Python, APIs, SDKs, Webhooks, Kibana, BigQuery, Datadog

Senior Product Specialist — Squarespace (2019-2021)
- Partnered with product/engineering on platform improvements
- Developed JavaScript, React Native solutions for complex customer issues

Developer Support Specialist — Acuity Scheduling (2017-2019)
- Resolved complex API and webhook integration failures
- Designed multi-step integration guides for developer self-service

Projects:
- Solune: Full-stack mobile app built with React Native (OpenAI, Supabase,
  RevenueCat, Apple Sign-In/SSO, Supabase Auth)
  Multi-step AI inference workflows, real-time data sync, production operation

Skills: Python, JavaScript, React Native, SQL, APIs, Docker, Kubernetes, AWS,
        Datadog, Kibana, BigQuery, OpenAI, Claude/Anthropic, Claude Code,
        Cursor, Supabase, LangGraph, LangSmith, Zendesk, Jira, Confluence,
        Salesforce
"""

# ── STATE ─────────────────────────────────────────────────────────────────────
class JobFitState(TypedDict):
    job_title: str
    company: str
    job_description: str
    requirements: List[str]
    fit_score: int
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    saved: bool

# ── LLM ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── NODES ─────────────────────────────────────────────────────────────────────
def extract_node(state: JobFitState) -> JobFitState:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="""Extract the key requirements from this job description.
Return ONLY a JSON array of strings, no other text.
Focus on: technical skills, years of experience, tools, soft skills, location requirements."""),
        HumanMessage(content=state["job_description"])
    ])
    try:
        requirements = json.loads(response.content)
    except:
        requirements = [response.content]
    return {"requirements": requirements}


def score_node(state: JobFitState) -> JobFitState:
    llm = get_llm()
    requirements_list = "\n".join(f"- {r}" for r in state["requirements"])
    response = llm.invoke([
        SystemMessage(content="""You are a technical recruiter analyzing candidate fit.
Given a resume and job requirements, return ONLY valid JSON in this exact format:
{
  "fit_score": <integer 0-100>,
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2", "gap 3"]
}
Scoring: 80-100 strong fit, 60-79 good fit, 40-59 moderate, 0-39 weak.
Be honest and specific. Reference actual items from the resume."""),
        HumanMessage(content=f"RESUME:\n{MY_RESUME}\n\nJOB REQUIREMENTS:\n{requirements_list}")
    ])
    try:
        result = json.loads(response.content)
        return {
            "fit_score": result.get("fit_score", 0),
            "strengths": result.get("strengths", []),
            "gaps": result.get("gaps", [])
        }
    except:
        return {"fit_score": 0, "strengths": [], "gaps": ["Could not parse result"]}


def recommend_node(state: JobFitState) -> JobFitState:
    if not state["gaps"]:
        return {"recommendations": ["No significant gaps — strong fit!"]}
    llm = get_llm()
    gaps_list = "\n".join(f"- {g}" for g in state["gaps"])
    response = llm.invoke([
        SystemMessage(content="""You are a career coach. Given skill gaps, provide specific actionable recommendations.
Return ONLY a JSON array of strings. Name specific resources, projects, or actions."""),
        HumanMessage(content=f"Job: {state['job_title']} at {state['company']}\n\nGaps:\n{gaps_list}")
    ])
    try:
        return {"recommendations": json.loads(response.content)}
    except:
        return {"recommendations": [response.content]}


def save_node(state: JobFitState) -> JobFitState:
    return {"saved": False}


# ── GRAPH ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_graph():
    graph = StateGraph(JobFitState)
    graph.add_node("extract", extract_node)
    graph.add_node("score", score_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "score")
    graph.add_edge("score", "recommend")
    graph.add_edge("recommend", "save")
    graph.add_edge("save", END)
    return graph.compile()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Fit Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    .score-box {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
    }
    .score-number {
        font-size: 72px;
        font-weight: 800;
        line-height: 1;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Job Fit Analyzer")
st.markdown("*Paste in a job description and see how well your profile matches.*")
st.divider()

# Input
col1, col2 = st.columns([1, 2])

with col1:
    job_title = st.text_input("Job Title", placeholder="e.g. Deployed Engineer")
    company = st.text_input("Company", placeholder="e.g. LangChain")

with col2:
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the full job description here...",
        height=200
    )

analyze_btn = st.button("Analyze Fit →", type="primary", use_container_width=True)

# Analysis
if analyze_btn:
    if not job_title or not company or not job_description:
        st.error("Please fill in all fields before analyzing.")
    else:
        agent = build_graph()

        with st.spinner("Analyzing your fit..."):
            # Progress steps
            progress = st.empty()
            progress.markdown("⚙️ *Extracting requirements...*")

            result = agent.invoke({
                "job_title": job_title,
                "company": company,
                "job_description": job_description,
                "requirements": [],
                "fit_score": 0,
                "strengths": [],
                "gaps": [],
                "recommendations": [],
                "saved": False
            })

            progress.empty()

        # Score display
        score = result["fit_score"]
        if score >= 80:
            color = "#4ade80"
            label = "Strong Fit"
            emoji = "🟢"
        elif score >= 60:
            color = "#facc15"
            label = "Good Fit"
            emoji = "🟡"
        elif score >= 40:
            color = "#fb923c"
            label = "Moderate Fit"
            emoji = "🟠"
        else:
            color = "#f87171"
            label = "Weak Fit"
            emoji = "🔴"

        st.divider()

        # Score + summary row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="score-box">
                <div class="score-number" style="color:{color}">{score}</div>
                <div style="font-size:14px;color:#6b6b80;margin-top:8px;">out of 100</div>
                <div style="font-size:18px;font-weight:600;color:{color};margin-top:8px;">{emoji} {label}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("**✅ Strengths**")
            for s in result["strengths"]:
                st.markdown(f"- {s}")

        with c3:
            st.markdown("**⚠️ Gaps**")
            for g in result["gaps"]:
                st.markdown(f"- {g}")

        st.divider()

        # Requirements extracted
        with st.expander("📋 Requirements extracted from job description"):
            for r in result["requirements"]:
                st.markdown(f"- {r}")

        # Recommendations
        st.markdown("### 📌 How to close the gaps")
        for i, rec in enumerate(result["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

        st.divider()
        st.caption("Traces available in LangSmith → smith.langchain.com")