import os
import requests
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

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

# 3. Model & Agent Setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
agent = create_react_agent(llm, tools)

# 4. Input & Output Schemas for LangServe
class AgentInput(BaseModel):
    input: str = Field(..., description="User query for the agent")

class AgentOutput(BaseModel):
    output: str = Field(..., description="Final text output")

# 5. Custom Runnable to handle Sync/Async Execution for ASGI
class AgentRunner(Runnable[AgentInput, AgentOutput]):
    def invoke(self, input: AgentInput, config: RunnableConfig | None = None) -> AgentOutput:
        query = input.input if isinstance(input, AgentInput) else str(input)
        res = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = res.get("messages", [])
        text = str(messages[-1].content) if messages and hasattr(messages[-1], "content") else "No response."
        return AgentOutput(output=text)

    async def ainvoke(self, input: AgentInput, config: RunnableConfig | None = None) -> AgentOutput:
        query = input.input if isinstance(input, AgentInput) else str(input)
        res = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        messages = res.get("messages", [])
        text = str(messages[-1].content) if messages and hasattr(messages[-1], "content") else "No response."
        return AgentOutput(output=text)

# 6. FastAPI Setup
app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description="Placement assistance agent"
)

add_routes(
    app,
    AgentRunner(),
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
