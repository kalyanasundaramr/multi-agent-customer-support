import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ANSWERS_FILE = BASE_DIR / "answers.txt"

load_dotenv(ENV_FILE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# --------------------------------------------------
# 2. Function-based file-writing tool
# --------------------------------------------------

@tool("save_support_record")
def save_support_record(
    query: str,
    assistant_answer: str,
    web_search_answer: str,
) -> str:
    """Save the original customer query and both agent answers to answers.txt.

    Args:
        query: The original customer query.
        assistant_answer: The complete answer from the Assistant agent.
        web_search_answer: The complete answer from the Web Search Assistant.
    """
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    record = (
        "\n"
        + "=" * 72
        + "\n"
        + f"Timestamp: {timestamp}\n\n"
        + "CUSTOMER QUERY\n"
        + f"{query}\n\n"
        + "ASSISTANT ANSWER\n"
        + f"{assistant_answer}\n\n"
        + "WEB SEARCH ANSWER\n"
        + f"{web_search_answer}\n"
        + "=" * 72
        + "\n"
    )

    try:
        with ANSWERS_FILE.open("a", encoding="utf-8") as file:
            file.write(record)

        return (
            "Successfully saved the customer query and both "
            f"answers to {ANSWERS_FILE.name}."
        )
    except OSError as exc:
        logger.exception("Could not save the support record.")
        return f"ERROR: Could not save the record: {exc}"


# --------------------------------------------------
# 3. Function to build exactly three agents and three tasks
# --------------------------------------------------

def build_support_crew():
    required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY"]
    missing_keys = [
        key for key in required_keys
        if not os.getenv(key, "").strip()
    ]

    if missing_keys:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing_keys)
            + ". Check your .env file."
        )

    web_search_tool = SerperDevTool()

    # Agent 1: Direct answer from its own knowledge.
    assistant = Agent(
        role="Customer Support Assistant",
        goal=(
            "Answer the customer's query clearly and accurately "
            "using your own knowledge, without web-search tools."
        ),
        backstory=(
            "You are a helpful customer support professional. "
            "You explain concepts in plain language, distinguish "
            "facts from uncertainty, and avoid inventing details."
        ),
        llm=MODEL_NAME,
        tools=[],
        allow_delegation=False,
        verbose=True,
    )

    # Agent 2: Search the web and prepare an independent answer.
    web_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Research the customer's query using web search and "
            "provide an evidence-based answer with useful source URLs."
        ),
        backstory=(
            "You are a careful web researcher. You use search results "
            "to support your answer, prefer trustworthy sources, "
            "and clearly identify uncertainty or conflicting findings."
        ),
        llm=MODEL_NAME,
        tools=[web_search_tool],
        allow_delegation=False,
        verbose=True,
    )

    # Agent 3: Save the original query and both complete answers.
    entry_agent = Agent(
        role="Customer Support Record Keeper",
        goal=(
            "Save the original query and both preceding agent answers "
            "to a text file, confirming whether the save succeeded."
        ),
        backstory=(
            "You are a meticulous records assistant. You preserve "
            "the supplied content accurately and use the dedicated "
            "file-writing tool instead of claiming an unsaved record."
        ),
        llm=MODEL_NAME,
        tools=[save_support_record],
        allow_delegation=False,
        verbose=True,
    )

    # Task 1: Direct answer.
    assistant_task = Task(
        description=(
            "Answer the following customer query directly from your "
            "own knowledge. Do not use web search. Explain the answer "
            "clearly, state relevant limitations, and avoid unsupported "
            "claims.\n\n"
            "Customer query:\n{query}"
        ),
        expected_output=(
            "A clear, self-contained answer to the customer's query."
        ),
        agent=assistant,
    )

    # Task 2: Web research and answer.
    web_search_task = Task(
        description=(
            "Research the original customer query using your web-search "
            "tool. Form your answer primarily from the search results. "
            "Include useful source URLs and distinguish verified "
            "information from uncertainty. Do not simply copy the "
            "first agent's answer.\n\n"
            "Customer query:\n{query}\n\n"
            "The preceding Assistant answer is available as context. "
            "Use it to understand the question, but independently "
            "verify important claims where possible."
        ),
        expected_output=(
            "A researched answer with supporting source URLs, "
            "relevant findings, and any important limitations."
        ),
        agent=web_assistant,
        context=[assistant_task],
    )

    # Task 3: Write both previous task outputs to answers.txt.
    entry_task = Task(
        description=(
            "Save this interaction using the save_support_record tool.\n\n"
            "Original customer query:\n{query}\n\n"
            "Use the complete output of the first task as "
            "assistant_answer and the complete output of the second "
            "task as web_search_answer. Preserve both answers faithfully. "
            "Call the tool with all three values. If the tool reports "
            "a saving error, report the failure honestly. Do not claim "
            "that a file was saved unless the tool confirms success."
        ),
        expected_output=(
            "A confirmation from the file-writing tool, including "
            "the filename, or an explicit report that saving failed."
        ),
        agent=entry_agent,
        context=[assistant_task, web_search_task],
    )

    crew = Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[assistant_task, web_search_task, entry_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew, assistant_task, web_search_task, entry_task


# --------------------------------------------------
# 4. Streamlit user interface
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Customer Support",
    page_icon="💬",
    layout="wide",
)

st.title("Multi-Agent Customer Support")
st.caption(
    "CrewAI | Sequential Agents | Web Research | Persistent Records"
)

st.markdown(
    """
    Submit a question to receive two perspectives:

    - **Assistant:** an answer based on the model's own knowledge.
    - **Web Search Assistant:** an answer informed by web research.

    The Entry Agent saves the query and both answers to `answers.txt`.
    """
)

with st.sidebar:
    st.header("System information")
    st.write("**Framework:** CrewAI")
    st.write("**Execution:** Sequential")
    st.write("**Agents:** 3")
    st.write("**Model:**", MODEL_NAME)
    st.write("**Record file:** `answers.txt`")
    st.caption(
        "The web-search agent requires a valid Serper API key. "
        "AI responses may contain errors; verify important information."
    )

if "support_result" not in st.session_state:
    st.session_state.support_result = None

with st.form("support_query_form"):
    query = st.text_area(
        "Enter your customer support query",
        placeholder="For example: How do I reset my password?",
        height=120,
    )

    submitted = st.form_submit_button(
        "Run Support Crew",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        st.session_state.support_result = None

        try:
            with st.spinner(
                "Running Agent 1 → Agent 2 → Agent 3. "
                "Web research may take a little time..."
            ):
                crew, assistant_task, web_task, entry_task = (
                    build_support_crew()
                )

                crew.kickoff(inputs={"query": query.strip()})

                assistant_output = (
                    assistant_task.output.raw
                    if assistant_task.output
                    else ""
                )

                web_output = (
                    web_task.output.raw
                    if web_task.output
                    else ""
                )

                entry_output = (
                    entry_task.output.raw
                    if entry_task.output
                    else "No Entry Agent output was returned."
                )

                st.session_state.support_result = {
                    "query": query.strip(),
                    "assistant": str(assistant_output),
                    "web": str(web_output),
                    "entry": str(entry_output),
                }

        except Exception:
            logger.exception("Support crew execution failed.")
            st.error(
                "The support crew could not complete this request. "
                "Check your API credentials, network connection, "
                "terminal logs, and installed package compatibility."
            )

result = st.session_state.support_result

if result:
    st.divider()
    st.subheader("Support Results")
    st.caption("Query: " + result["query"])

    tab_assistant, tab_web, tab_entry = st.tabs(
        [
            "1. Assistant Answer",
            "2. Web Search Answer",
            "3. Save Status",
        ]
    )

    with tab_assistant:
        st.markdown(result["assistant"])

    with tab_web:
        st.markdown(result["web"])

    with tab_entry:
        st.markdown(result["entry"])

        if ANSWERS_FILE.exists():
            st.success(
                "The answers file exists at: " + str(ANSWERS_FILE)
            )
            st.caption(
                "The Entry Agent's tool confirmation above is the "
                "primary indication that this interaction was saved."
            )
        else:
            st.warning(
                "answers.txt does not exist yet. Verify the Entry "
                "Agent's tool execution."
            )

st.divider()
st.caption(
    "Buildathon project | API credentials are loaded from .env "
    "and must never be committed to source control."
)
