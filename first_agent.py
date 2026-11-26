import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

# 1) الـ LLM (حط الموديل اللي معاك API key بتاعه)
llm = LLM(model="gemini/gemini-3.1-flash-lite")   # الـ key بيتحط في ملف .env باسم GEMINI_API_KEY

# 2) الـ Tool: function عادية + decorator
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

import io, contextlib, traceback

@tool("run_pandas_code")
def run_pandas_code(code: str) -> str:
    """Runs Python pandas code on the dataset. The data is already loaded
    as a DataFrame named `df` (pandas is available as `pd`).
    Always use print() to show results. Returns the printed output or the error."""
    df = pd.read_csv("data.csv")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"df": df, "pd": pd})
    except Exception:
        return "ERROR:\n" + traceback.format_exc(limit=1)
    out = buffer.getvalue()
    return out[:3000] if out else "No output. Did you forget print()?"

analyst = Agent(
    role="Senior Data Analyst",
    goal="Answer questions about a dataset using real computed numbers",
    backstory="You never guess numbers. First inspect the data, then write "
              "pandas code to compute the answer, and fix your code if it errors.",
    tools=[csv_summary, run_pandas_code],
    llm=llm,
    max_iter=8,          # أقصى عدد خطوات (حماية من الـ loops اللانهائية)
    verbose=True,
)

task = Task(
    description="Dataset: data.csv. Answer this question: {question}",
    expected_output="A short answer with the exact numbers, and one line "
                    "explaining how you computed it.",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], verbose=True)
result = crew.kickoff(inputs={"question": "What is the average spending score by gender?"})
print(result)
