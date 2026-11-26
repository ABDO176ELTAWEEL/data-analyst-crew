import io
import os
import re
import time
import tempfile
import contextlib
import traceback

import matplotlib
matplotlib.use("Agg")  # no GUI window: required when running inside a web app
import matplotlib.pyplot as plt

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

MAX_FILE_MB = 5
MAX_QUESTION_CHARS = 300

llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Safety net for exec(): NOT a real sandbox.
BLOCKED = [
    "import os", "import sys", "subprocess", "open(", "eval(", "exec(",
    "__", "shutil", "requests", "socket", "to_csv", "remove(", "rmdir",
    "system(", "getattr", "globals", "locals",
]


def build_crew(data_path: str, chart_path: str) -> Crew:
    """Builds a fresh crew for ONE uploaded file.
    Tools close over data_path / chart_path, so runs never share state."""

    @tool("csv_summary")
    def csv_summary(path: str) -> str:
        """Returns shape, column types, missing values and first rows of the dataset.
        Pass the file name as given in the task."""
        df = pd.read_csv(data_path)  # `path` is ignored on purpose
        return (
            f"Shape: {df.shape}\n\n"
            f"Dtypes:\n{df.dtypes}\n\n"
            f"Missing values:\n{df.isna().sum()}\n\n"
            f"Head:\n{df.head()}"
        )

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
        df = pd.read_csv(data_path)
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, {"df": df, "pd": pd})
        except Exception:
            return "ERROR:\n" + traceback.format_exc(limit=1)
        out = buffer.getvalue()
        return out[:3000] if out else "No output. Did you forget print()?"

    @tool("make_chart")
    def make_chart(code: str) -> str:
        """Creates ONE chart as a PNG image using matplotlib.
        The data is loaded as a DataFrame named `df`; `pd` is pandas and `plt`
        is matplotlib.pyplot. Write code that builds the chart with plt
        (bar, line, hist, scatter...) and always add a title and axis labels.
        Do NOT call plt.show() or plt.savefig(): the tool saves the figure itself.
        Returns a success message or the error message."""
        for word in BLOCKED + ["savefig", "plt.show"]:
            if word in code:
                return (f"ERROR: '{word}' is not allowed. "
                        "Only build the chart with plt; the tool saves it.")
        df = pd.read_csv(data_path)
        plt.close("all")
        try:
            plt.figure(figsize=(7, 4.5))
            exec(code, {"df": df, "pd": pd, "plt": plt})
            plt.tight_layout()
            plt.savefig(chart_path, dpi=120)
        except Exception:
            plt.close("all")
            return "ERROR:\n" + traceback.format_exc(limit=1)
        plt.close("all")
        return "Chart saved successfully."

    # ---------------- Agents ----------------
    analyst = Agent(
        role="Senior Data Analyst",
        goal="Compute accurate answers from the data",
        backstory=(
            "You inspect the data first, then compute with pandas. "
            "Never guess numbers. If your code fails, read the error and fix it. "
            "Always state every assumption you make (grouping, binning, filtering). "
            "If the user gave explicit bins or filters, use them exactly."
        ),
        tools=[csv_summary, run_pandas_code],
        llm=llm, max_iter=8, max_execution_time=120, max_retry_limit=2,
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
        llm=llm, max_iter=6, max_execution_time=120, max_retry_limit=2,
        verbose=True,
    )
    visualizer = Agent(
        role="Data Visualizer",
        goal="Create one clear chart that supports the verified findings",
        backstory=(
            "You make simple, readable charts (bar, line, histogram). "
            "You chart only what the verified results show, with the same "
            "grouping the analyst used, a title, and labeled axes."
        ),
        tools=[make_chart],
        llm=llm, max_iter=5, max_execution_time=90, max_retry_limit=2,
        verbose=True,
    )
    writer = Agent(
        role="Report Writer",
        goal="Explain the verified findings clearly to a non-technical reader",
        backstory=(
            "You only use numbers that the reviewer confirmed. "
            "You never add new numbers and you never claim a trend or cause "
            "that the verified numbers do not show. You always mention the assumptions."
        ),
        llm=llm, max_iter=4, max_execution_time=90, max_retry_limit=2,
        verbose=True,
    )

    # ---------------- Tasks ----------------
    t1 = Task(
        description="Dataset: the uploaded CSV file. Answer this question: {question}",
        expected_output=(
            "The answer with exact numbers, the pandas code used, and a line "
            "'Assumptions:' listing any grouping, binning or filtering choices."
        ),
        agent=analyst,
    )
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
    t_chart = Task(
        description=(
            "Using the verified findings, create ONE chart that helps answer: "
            "{question}. Reuse the same grouping/bins as the analyst's pandas code. "
            "If the answer is a single number and a chart would not help, "
            "do not create a chart."
        ),
        expected_output=(
            "One sentence describing the chart you created, "
            "or 'No chart needed'."
        ),
        agent=visualizer,
        context=[t1, t2],
    )

    def numbers_are_verified(output):
        report = output.raw
        verified = t2.output.raw if t2.output else ""
        numbers = re.findall(r"\d+(?:\.\d+)?", report)
        bad = [n for n in numbers if len(n) > 1 and n not in verified]
        if bad:
            return (False,
                    f"These numbers are not in the verified results: {bad}. "
                    "Remove them or use only verified numbers.")
        return (True, output)

    t3 = Task(
        description=(
            "Write a short report (under 150 words) answering {question} "
            "using only the verified numbers. Mention the assumptions used."
        ),
        expected_output=(
            "A short markdown report with a title, the answer, the assumptions, "
            "and one insight that is directly supported by the verified numbers."
        ),
        agent=writer,
        context=[t1, t2],
        guardrail=numbers_are_verified,
        guardrail_max_retries=2,
    )

    return Crew(
        agents=[analyst, reviewer, visualizer, writer],
        tasks=[t1, t2, t_chart, t3],
        max_rpm=10,
        verbose=True,
    )


def run_with_retry(crew: Crew, inputs: dict, tries: int = 3):
    for i in range(tries):
        try:
            return crew.kickoff(inputs=inputs)
        except Exception as e:
            wait = 2 ** (i + 1)
            print(f"Attempt {i + 1} failed: {type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("All retries failed")


def analyze(file_path, question):
    """Returns (report_markdown, chart_image_path_or_None)."""
    if not file_path:
        return "⚠️ Please upload a CSV file first.", None
    question = (question or "").strip()
    if not question:
        return "⚠️ Please type a question.", None
    if len(question) > MAX_QUESTION_CHARS:
        return f"⚠️ Question is too long (max {MAX_QUESTION_CHARS} characters).", None
    if os.path.getsize(file_path) / 1024 / 1024 > MAX_FILE_MB:
        return f"⚠️ File is too large (max {MAX_FILE_MB} MB).", None
    try:
        pd.read_csv(file_path, nrows=5)
    except Exception as e:
        return f"⚠️ Could not read this file as a CSV: {e}", None

    # Each request gets its own folder, so charts never overwrite each other.
    chart_path = os.path.join(tempfile.mkdtemp(prefix="crew_chart_"), "chart.png")

    try:
        crew = build_crew(file_path, chart_path)
        result = run_with_retry(crew, {"question": question})
        report = getattr(result, "raw", str(result))
    except Exception as e:
        return (f"❌ Something went wrong ({type(e).__name__}). "
                "Please try again in a minute."), None

    has_chart = os.path.exists(chart_path) and os.path.getsize(chart_path) > 0
    return report, (chart_path if has_chart else None)


with gr.Blocks(title="Data Analyst Crew") as demo:
    gr.Markdown(
        "# 📊 Data Analyst Crew\n"
        "Upload a CSV, ask a question in English. Four agents "
        "(analyst → reviewer → visualizer → writer) analyze it and give you "
        "a short report and a chart.\n\n"
        "*It can take 2–3 minutes. Always double-check the chart against the report.*"
    )
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
            question_in = gr.Textbox(
                label="Your question",
                placeholder="e.g. Which age group (18-25, 26-35, 36-50, 51+) "
                            "has the highest average spending score?",
                lines=3,
            )
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            report_out = gr.Markdown(label="Report")
            chart_out = gr.Image(label="Chart", type="filepath")

    run_btn.click(analyze, inputs=[file_in, question_in],
                  outputs=[report_out, chart_out])

if __name__ == "__main__":
    demo.launch()