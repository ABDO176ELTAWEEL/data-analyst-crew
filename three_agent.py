import io
import os
import re
import time
import contextlib
import traceback

import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

DATA_PATH = "data.csv"

# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------
llm = LLM(model="gemini/gemini-3.1-flash-lite")

# ---------------------------------------------------------------
# Tools
# ---------------------------------------------------------------
@tool("csv_summary")
def csv_summary(path: str) -> str:
    """Returns shape, column types, missing values and first rows of a CSV file."""
    df = pd.read_csv(path)
    return (
        f"Shape: {df.shape}\n\n"
        f"Dtypes:\n{df.dtypes}\n\n"
        f"Missing values:\n{df.isna().sum()}\n\n"
        f"Head:\n{df.head()}"
    )


# Guardrail (C): block dangerous code before exec().
# NOTE: this is a safety net, NOT a real sandbox.
BLOCKED = [
    "import os", "import sys", "subprocess", "open(", "eval(", "exec(",
    "__", "shutil", "requests", "socket", "to_csv", "remove(", "rmdir",
    "system(", "getattr", "globals", "locals",
]


@tool("run_pandas_code")
def run_pandas_code(code: str) -> str:
    """Runs Python pandas code on the dataset. The data is already loaded
    as a DataFrame named `df` (pandas is available as `pd`).
    Always use print() to show results.
    Returns the printed output or the error message."""
    for word in BLOCKED:
        if word in code:
            return (f"ERROR: '{word}' is not allowed. "
                    "Use only pandas operations on `df`.")

    df = pd.read_csv(DATA_PATH)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"df": df, "pd": pd})
    except Exception:
        return "ERROR:\n" + traceback.format_exc(limit=1)

    out = buffer.getvalue()
    return out[:3000] if out else "No output. Did you forget print()?"


# ---------------------------------------------------------------
# Agents  (Guardrail A: limits on every agent)
# ---------------------------------------------------------------
analyst = Agent(
    role="Senior Data Analyst",
    goal="Compute accurate answers from the data",
    backstory=(
        "You inspect the data first, then compute with pandas. "
        "Never guess numbers. If your code fails, read the error and fix it. "
        "Always state every assumption you make (grouping, binning, filtering)."
    ),
    tools=[csv_summary, run_pandas_code],
    llm=llm,
    max_iter=8,
    max_execution_time=120,
    max_retry_limit=2,
    verbose=True,
)

reviewer = Agent(
    role="Data Reviewer",
    goal="Catch wrong numbers and wrong assumptions before they reach the report",
    backstory=(
        "You are skeptical. You recompute every number yourself with pandas "
        "using a different approach, and you question the analyst's assumptions."
    ),
    tools=[run_pandas_code],
    llm=llm,
    max_iter=6,
    max_execution_time=120,
    max_retry_limit=2,
    verbose=True,
)

writer = Agent(
    role="Report Writer",
    goal="Explain the verified findings clearly to a non-technical reader",
    backstory=(
        "You only use numbers that the reviewer confirmed. "
        "You never add new numbers. You always mention the assumptions."
    ),
    llm=llm,
    max_iter=4,
    max_execution_time=90,
    max_retry_limit=2,
    verbose=True,
)

# ---------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------
t1 = Task(
    description="Dataset: data.csv. Answer this question: {question}",
    expected_output=(
        "The answer with exact numbers, the pandas code used, and a line "
        "'Assumptions:' listing any grouping, binning or filtering choices."
    ),
    agent=analyst,
)

# Guardrail (D): the reviewer must also judge the assumptions.
t2 = Task(
    description=(
        "Recompute the analyst's numbers independently using pandas. "
        "Also check whether the analyst's assumptions are reasonable. "
        "State clearly: CONFIRMED or WRONG, and give the correct numbers if wrong."
    ),
    expected_output=(
        "A verdict (CONFIRMED / WRONG), the final verified numbers, "
        "and a line 'Assumptions check:' with your opinion on the assumptions."
    ),
    agent=reviewer,
    context=[t1],
)


# Guardrail (B): every number in the report must exist in the reviewer output.
def numbers_are_verified(output):
    report = output.raw
    verified = t2.output.raw if t2.output else ""
    numbers = re.findall(r"\d+(?:\.\d+)?", report)
    bad = [n for n in numbers if len(n) > 1 and n not in verified]
    if bad:
        return (
            False,
            f"These numbers are not in the verified results: {bad}. "
            "Remove them or use only verified numbers.",
        )
    return (True, output)


t3 = Task(
    description=(
        "Write a short report (under 150 words) answering {question} "
        "using only the verified numbers. Mention the assumptions used."
    ),
    expected_output=(
        "A short markdown report with a title, the answer, the assumptions, "
        "and one insight."
    ),
    agent=writer,
    context=[t1, t2],
    output_file="report.md",
    guardrail=numbers_are_verified,
    guardrail_max_retries=2,
)

# ---------------------------------------------------------------
# Crew  (max_rpm protects the free-tier rate limit)
# ---------------------------------------------------------------
crew = Crew(
    agents=[analyst, reviewer, writer],
    tasks=[t1, t2, t3],
    max_rpm=10,
    verbose=True,
)


# Retry with exponential backoff for API errors (503, rate limits...)
def run_with_retry(crew, inputs, tries=4):
    for i in range(tries):
        try:
            return crew.kickoff(inputs=inputs)
        except Exception as e:
            wait = 2 ** (i + 1)
            print(f"Attempt {i + 1} failed: {type(e).__name__}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("All retries failed")


if __name__ == "__main__":
    result = run_with_retry(
        crew,
        {"question": "Which age group has the highest average spending score?"},
    )
    print(result)