import streamlit as st
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from pyairtable import Api

load_dotenv()

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

def parse_json_response(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())

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

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_node(state: JobFitState) -> JobFitState:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="""Extract the key requirements from this job description.
Return ONLY a raw JSON array of strings, no markdown, no code fences, no explanation.
If the input is not a real job description, return an empty array: []"""),
        HumanMessage(content=state["job_description"])
    ])
    try:
        requirements = parse_json_response(response.content)
        if not isinstance(requirements, list):
            requirements = []
    except:
        requirements = []
    return {"requirements": requirements}

def score_node(state: JobFitState) -> JobFitState:
    if not state["requirements"]:
        return {"fit_score": 0, "strengths": [], "gaps": ["This does not appear to be a valid job description."]}
    llm = get_llm()
    requirements_list = "\n".join(f"- {r}" for r in state["requirements"])
    response = llm.invoke([
        SystemMessage(content="""You are a strict technical recruiter. Score ONLY based on how well the resume matches THIS SPECIFIC JOB.
Return ONLY raw JSON, no markdown, no code fences:
{"fit_score": <0-100>, "strengths": ["..."], "gaps": ["..."]}
Scoring: 80-100 strong match, 60-79 good, 40-59 moderate, 0-39 weak.
An unrelated job must score low. Be honest and specific."""),
        HumanMessage(content=f"RESUME:\n{MY_RESUME}\n\nJOB REQUIREMENTS:\n{requirements_list}")
    ])
    try:
        result = parse_json_response(response.content)
        return {
            "fit_score": int(result.get("fit_score", 0)),
            "strengths": result.get("strengths", []),
            "gaps": result.get("gaps", [])
        }
    except:
        return {"fit_score": 0, "strengths": [], "gaps": ["Could not parse result"]}

def recommend_node(state: JobFitState) -> JobFitState:
    if not state["gaps"] or state["fit_score"] == 0:
        return {"recommendations": []}
    llm = get_llm()
    gaps_list = "\n".join(f"- {g}" for g in state["gaps"])
    response = llm.invoke([
        SystemMessage(content="""You are a career coach. Give specific actionable recommendations to close these gaps.
Return ONLY a raw JSON array of strings, no markdown, no code fences.
Name actual courses, tools, or projects."""),
        HumanMessage(content=f"Job: {state['job_title']} at {state['company']}\n\nGaps:\n{gaps_list}")
    ])
    try:
        recs = parse_json_response(response.content)
        if not isinstance(recs, list):
            recs = [str(recs)]
        return {"recommendations": recs}
    except:
        return {"recommendations": [response.content]}

def save_node(state: JobFitState) -> JobFitState:
    try:
        token = os.environ.get("AIRTABLE_TOKEN")
        base_id = os.environ.get("AIRTABLE_BASE_ID")
        if not token or not base_id:
            return {"saved": False}

        api = Api(token)
        table = api.table(base_id, "Applications")
        table.create({
            "Job Title": state["job_title"],
            "Company": state["company"],
            "Fit Score": state["fit_score"],
            "Strengths": "\n".join(state["strengths"]),
            "Gaps": "\n".join(state["gaps"]),
            "Recommendations": "\n".join(state["recommendations"]),
            "Analyzed At": datetime.utcnow().strftime("%Y-%m-%d")
        })
        return {"saved": True}
    except Exception as e:
        print(f"Airtable save failed: {e}")
        return {"saved": False}

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

st.set_page_config(page_title="Job Fit Analyzer", page_icon="🎯", layout="wide")

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
    .score-number { font-size: 72px; font-weight: 800; line-height: 1; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Job Fit Analyzer")
st.markdown("*Paste in a job description and see how well your profile matches.*")
st.divider()

col1, col2 = st.columns([1, 2])
with col1:
    job_title = st.text_input("Job Title", placeholder="e.g. Deployed Engineer")
    company = st.text_input("Company", placeholder="e.g. LangChain")
with col2:
    job_description = st.text_area("Job Description", placeholder="Paste the full job description here...", height=200)

analyze_btn = st.button("Analyze Fit →", type="primary", use_container_width=True)

if analyze_btn:
    if not job_title or not company or not job_description:
        st.error("Please fill in all fields before analyzing.")
    else:
        agent = build_graph()
        with st.spinner("Analyzing your fit..."):
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

        score = result["fit_score"]
        if score >= 80:
            color, label, emoji = "#4ade80", "Strong Fit", "🟢"
        elif score >= 60:
            color, label, emoji = "#facc15", "Good Fit", "🟡"
        elif score >= 40:
            color, label, emoji = "#fb923c", "Moderate Fit", "🟠"
        else:
            color, label, emoji = "#f87171", "Weak Fit", "🔴"

        st.divider()
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

        if result["requirements"]:
            with st.expander("📋 Requirements extracted from job description"):
                for r in result["requirements"]:
                    st.markdown(f"- {r}")

        if result["recommendations"]:
            st.markdown("### 📌 How to close the gaps")
            for i, rec in enumerate(result["recommendations"], 1):
                st.markdown(f"**{i}.** {rec}")

        # Airtable save status
        if result.get("saved"):
            st.success("✅ Saved to Airtable")
        
        st.divider()
        st.caption("Traces available in LangSmith → smith.langchain.com")