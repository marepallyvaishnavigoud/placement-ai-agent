%%writefile app.py
import os
import requests
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Define Tools
@tool
def job_search(role: str) -> str:
    """Searches for recent campus or entry-level job roles and required skills."""
    return f"Found 3 openings for {role}: 1. SDE-1 (Requires Python, DSA, SQL). 2. AI Engineer (Requires PyTorch, LangChain, FastAPI)."

@tool
def skill_gap_analysis(resume_text: str, target_role: str) -> str:
    """Compares student skills against target role requirement."""
    return f"Skill gaps for {target_role}: Lacks hands-on experience in Docker, System Design, REST APIs."

@tool
def project_recommendation(skill_gap: str) -> str:
    """Suggests relevant projects based on skill gaps."""
    return "Recommended Project: Build a 'Placement-Ready AI Agent' using LangChain & FastAPI."

@tool
def github_evaluator(username: str) -> str:
    """Evaluates public GitHub profile activity and repositories."""
    res = requests.get(f"https://api.github.com/users/{username}/repos")
    if res.status_code == 200:
        repos = res.json()
        repo_names = [r['name'] for r in repos[:5]]
        return f"GitHub User '{username}' repos: {', '.join(repo_names)}."
    return f"Could not fetch profile for {username}."

tools = [job_search, skill_gap_analysis, project_recommendation, github_evaluator]

# App Setup
llm = ChatGoogleGenerativeAI(model="gemini-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description="Placement assistance agent built with LangChain and LangServe"
)

# Expose endpoint via LangServe
add_routes(app, llm_with_tools, path="/agent")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
