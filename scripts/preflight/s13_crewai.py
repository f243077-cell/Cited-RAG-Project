"""Step 13 (LIVE, 1 to 3 requests): one CrewAI agent and task on the real service, compare usage with requests sent."""

import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from common import count_request, load_env, result, set_api_key_env, stop_on_quota

from crewai import LLM, Agent, Crew, Process, Task
from pydantic import BaseModel


class Capital(BaseModel):
    city: str
    country: str


env = load_env()
set_api_key_env(env)
model = env["MODEL_FAST"]
agent = Agent(
    role="Geographer",
    goal="Answer geography questions",
    backstory="You know capitals.",
    llm=LLM(model=f"openrouter/{model}", max_tokens=1000),
    allow_delegation=False,
    verbose=False,
    max_iter=1,
    max_execution_time=60,
)
task = Task(
    description="What is the capital of {country}?",
    expected_output="City and country",
    agent=agent,
    output_pydantic=Capital,
)
count_request(3)  # up to 3 HTTP attempts: reserve them all, then read the real figure from the usage below
try:
    out = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
        inputs={"country": "France"}
    )
    result(out.pydantic is not None, f"{model}: {out.pydantic}; token_usage={out.token_usage}")
except Exception as exc:  # noqa: BLE001
    stop_on_quota(exc)
    result(False, f"{model}: {type(exc).__name__}: {str(exc)[:300]}")
