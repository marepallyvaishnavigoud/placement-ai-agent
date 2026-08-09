import os
import requests
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# 1. Environment Setup
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# 2. Define Tools
@tool
def job_search(role: str) -> str:
    """Searches for recent campus or entry-level job roles and required skills."""
    return f"Found openings for {role}: 1. SDE-1 (Requires Python, DSA, SQL). 2. AI Engineer (Requires PyTorch, LangChain, FastAPI)."

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
        return f"GitHub User '{username}' public repos: {', '.join(repo_names)}."
    return f"Could not fetch profile for {username}."

tools = [job_search, skill_gap_analysis, project_recommendation, github_evaluator]

# 3. Model & ReAct Agent Setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
agent = create_react_agent(llm, tools)

# 4. Agent Execution Function
def run_agent_and_extract_string(query_str: str) -> str:
    if isinstance(query_str, dict):
        query_str = query_str.get("input", str(query_str))
    
    # Run the ReAct agent graph
    result = agent.invoke({"messages": [{"role": "user", "content": str(query_str)}]})
    
    # Extract final text message
    messages = result.get("messages", [])
    if messages and hasattr(messages[-1], "content"):
        return str(messages[-1].content)
    return "No response generated."

# 5. LCEL Pipeline Construction
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

# Converting input dictionary directly into string output for LangServe UI
chain = prompt | RunnableLambda(lambda x: run_agent_and_extract_string(x.to_string()))

# 6. FastAPI App Setup
app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description="Placement assistance agent"
)

add_routes(
    app,
    chain,
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
