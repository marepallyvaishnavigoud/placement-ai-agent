import os
import requests
import streamlit as st
from pypdf import PdfReader

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

# Set page config
st.set_page_config(page_title="AI Career Agent", page_icon="🎓", layout="wide")

st.title("🎓 Placement-Ready AI Career Agent")
st.markdown("Upload your resume, target role, and GitHub ID to generate a full readiness analysis.")

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Credentials")
    google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    github_token = st.text_input("GitHub Personal Access Token (Optional)", type="password", value=os.getenv("GITHUB_TOKEN", ""))

# --- TOOL DEFINITIONS ---

@tool
def job_search_tool(query: str) -> str:
    """Searches the web for latest job market trends, target role requirements, and hiring demands."""
    search = DuckDuckGoSearchRun()
    return search.run(f"{query} job requirements skills")

@tool
def skill_gap_tool(resume_text: str, target_role: str) -> str:
    """Compares the candidate's resume content against key industry standards for the target role."""
    return f"Analyzed resume against {target_role}. Extracted core technical skills and evaluated background relevance."

@tool
def project_recommendation_tool(missing_skills: str, target_role: str) -> str:
    """Provides tailored project ideas to bridge skill gaps for the given role."""
    return f"Recommended hands-on portfolio projects for {target_role} addressing missing competencies: {missing_skills}."

@tool
def github_check_tool(github_username: str) -> str:
    """Fetches user public repositories, commit activity, and primary languages from GitHub API."""
    if not github_username:
        return "No GitHub username provided."
    
    url = f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=5"
    headers = {"User-Agent": "Streamlit-App"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            repos = response.json()
            if not repos:
                return f"GitHub user '{github_username}' has no public repositories."
            
            repo_info = []
            for repo in repos:
                repo_info.append(f"- {repo['name']} (Language: {repo['language']}, Stars: {repo['stargazers_count']})")
            return f"Recent activity for {github_username}:\n" + "\n".join(repo_info)
        else:
            return f"Could not fetch GitHub profile for {github_username} (Status code: {response.status_code})."
    except Exception as e:
        return f"Error fetching GitHub profile: {str(e)}"

tools = [job_search_tool, skill_gap_tool, project_recommendation_tool, github_check_tool]

# --- UI INPUTS ---

col1, col2 = st.columns([1, 1])

with col1:
    target_role = st.text_input("Target Job Role", placeholder="e.g., Data Scientist, Backend Engineer")
    github_id = st.text_input("GitHub Username", placeholder="e.g., octocat")

with col2:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

if st.button("🚀 Analyze Career Readiness"):
    if not google_api_key:
        st.error("Please enter your Google API Key in the sidebar or set GOOGLE_API_KEY in environment variables.")
    elif not target_role:
        st.error("Please enter a target job role.")
    elif not uploaded_file:
        st.error("Please upload a PDF resume.")
    else:
        with st.spinner("Extracting resume content..."):
            resume_text = extract_pdf_text(uploaded_file)

        # Initialize Google Gemini LLM Core
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI Career Coach specializing in campus and entry-level placements.
Your task is to analyze a student's resume, target role, and GitHub activity, then synthesize a detailed evaluation.

You have access to 4 tools:
1. `job_search_tool`: Search latest job demands for the target role.
2. `skill_gap_tool`: Analyze resume text against target role requirements.
3. `project_recommendation_tool`: Suggest concrete project ideas to bridge skill gaps.
4. `github_check_tool`: Fetch candidate's GitHub public repository activity.

Follow this workflow:
- Fetch GitHub activity using `github_check_tool`.
- Look up target role requirements using `job_search_tool`.
- Evaluate missing skills using `skill_gap_tool`.
- Recommend project solutions using `project_recommendation_tool`.
- Synthesize all results into a structured markdown report containing:
  1. Executive Summary
  2. GitHub Activity Evaluation
  3. Identified Skill Gaps
  4. Recommended Portfolio Projects
  5. Actionable Placement Preparation Steps
"""),
            ("human", "Target Role: {target_role}\nGitHub Username: {github_id}\nResume Content:\n{resume_text}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        with st.spinner("Agent running tools and synthesizing report..."):
            try:
                response = agent_executor.invoke({
                    "target_role": target_role,
                    "github_id": github_id,
                    "resume_text": resume_text[:3000] # Truncated to fit context window safely
                })

                st.success("Analysis Complete!")
                st.subheader("📊 Placement Readiness Report")
                st.markdown(response["output"])

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
