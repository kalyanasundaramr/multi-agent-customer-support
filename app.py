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
        verbose=False,
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
        verbose=False,
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
        verbose=False,
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
        verbose=False,
    )

    return crew, assistant_task, web_search_task, entry_task


# --------------------------------------------------
# 4. Modern Streamlit user interface
# --------------------------------------------------

st.set_page_config(
    page_title="SupportDesk AI | Customer Support",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17211d;
        --muted: #68756f;
        --green: #1f6b50;
        --green-dark: #154b39;
        --mint: #eaf5ef;
        --line: #e3eae5;
        --surface: #ffffff;
        --warm: #f7f9f6;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(204, 232, 215, .32), transparent 26rem),
            #f7f9f7;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1220px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f0f5f1;
        border-right: 1px solid #e0e9e2;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .brand-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 1.2rem;
    }
    .brand-mark {
        width: 42px; height: 42px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        background: #1f6b50; color: white; font-size: 23px;
        box-shadow: 0 7px 18px rgba(31,107,80,.18);
    }
    .brand-name { font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 800; color: #17211d; }
    .brand-sub { font-size: .76rem; color: #718078; margin-top: 1px; }
    .eyebrow {
        color: #267554; text-transform: uppercase; letter-spacing: .14em;
        font-size: .72rem; font-weight: 800; margin-bottom: .55rem;
    }
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #163f31 0%, #1f6b50 62%, #31805e 100%);
        border-radius: 25px; padding: 2.15rem 2.2rem;
        color: #fff; margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(24, 74, 54, .14);
    }
    .hero:after {
        content: ""; position: absolute; width: 280px; height: 280px;
        right: -70px; top: -125px; border-radius: 50%;
        border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 0 0 30px rgba(255,255,255,.035), 0 0 0 60px rgba(255,255,255,.025);
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.15; letter-spacing: -.04em; margin: 0 0 .8rem 0;
        color: #fff; max-width: 700px;
    }
    .hero p { color: rgba(255,255,255,.82); max-width: 680px; font-size: 1rem; line-height: 1.7; margin: 0; }
    .hero-chip {
        display: inline-flex; align-items: center; gap: 7px;
        border: 1px solid rgba(255,255,255,.22); background: rgba(255,255,255,.09);
        border-radius: 99px; padding: 6px 11px; font-size: .75rem;
        color: #f0fff5; margin-bottom: 1rem;
    }
    .section-heading {
        font-family: 'Manrope', sans-serif; font-weight: 800;
        font-size: 1.18rem; letter-spacing: -.025em; color: #17211d;
        margin: .35rem 0 .25rem 0;
    }
    .section-copy { color: #718078; font-size: .9rem; margin-bottom: 1rem; }
    .metric-card {
        background: rgba(255,255,255,.9); border: 1px solid #e4ebe5;
        border-radius: 17px; padding: 1rem 1.1rem; min-height: 105px;
    }
    .metric-label { color: #748078; font-size: .78rem; font-weight: 600; }
    .metric-value { color: #17211d; font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 800; margin-top: .25rem; }
    .metric-note { color: #2d7857; font-size: .75rem; margin-top: .15rem; }
    .panel {
        background: #fff; border: 1px solid #e3eae5; border-radius: 20px;
        padding: 1.2rem 1.3rem; margin-bottom: 1rem;
        box-shadow: 0 5px 18px rgba(27, 56, 40, .025);
    }
    .agent-pill {
        display: inline-block; background: #eaf5ef; color: #236849;
        border-radius: 99px; padding: 5px 10px; font-size: .73rem;
        font-weight: 700; margin-bottom: .55rem;
    }
    .muted { color: #718078; font-size: .86rem; }
    .stTextArea textarea {
        border-radius: 14px !important; border-color: #dce6de !important;
        background: #fcfdfc !important; font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #398361 !important;
        box-shadow: 0 0 0 1px #398361 !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 12px; font-weight: 700; min-height: 2.8rem;
        border: 1px solid #1f6b50;
    }
    .stFormSubmitButton > button[kind="primary"] {
        background: #1f6b50; color: white;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #e3eae5; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 10px 14px;
        font-weight: 700; color: #65746b;
    }
    .stTabs [aria-selected="true"] { color: #1f6b50 !important; }
    div[data-testid="stAlert"] { border-radius: 13px; }
    .footer-note { text-align: center; color: #87938c; font-size: .76rem; padding-top: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar brand and configuration.
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">✳</div>
          <div>
            <div class="brand-name">SupportDesk AI</div>
            <div class="brand-sub">Multi-agent help center</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Workspace")
    st.markdown("**Workflow**")
    st.caption("Ask → Research → Save")
    st.markdown("**AI model**")
    st.code(MODEL_NAME, language=None)
    st.markdown("**Agents online**")
    st.markdown(
        """
        <div style="display:flex;gap:8px;align-items:center;margin:.35rem 0">
          <span style="height:8px;width:8px;background:#2f9b67;border-radius:50%;display:inline-block"></span>
          <span style="font-size:.86rem">Knowledge assistant</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin:.35rem 0">
          <span style="height:8px;width:8px;background:#2f9b67;border-radius:50%;display:inline-block"></span>
          <span style="font-size:.86rem">Web research agent</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin:.35rem 0">
          <span style="height:8px;width:8px;background:#2f9b67;border-radius:50%;display:inline-block"></span>
          <span style="font-size:.86rem">Record keeper</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Record storage")
    st.caption(f"`{ANSWERS_FILE.name}`")
    st.caption("Each saved interaction includes a timestamp, the question, and both answers.")
    st.markdown("---")
    st.caption(
        "Credentials are loaded from `.env`. Never share or commit API keys."
    )

# Hero header.
st.markdown(
    """
    <div class="hero">
      <div class="hero-chip">✦ &nbsp; AI-POWERED CUSTOMER CARE</div>
      <h1>Good support starts with a clear answer.</h1>
      <p>Ask once. Get an answer from the assistant, a second perspective grounded in web research, and a saved record for your team — all in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Lightweight overview cards.
saved_count = 0
if ANSWERS_FILE.exists():
    try:
        saved_count = ANSWERS_FILE.read_text(encoding="utf-8").count("CUSTOMER QUERY")
    except OSError:
        saved_count = 0

m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">WORKFLOW</div><div class="metric-value">3 agents</div><div class="metric-note">Sequential hand-off</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">RESEARCH</div><div class="metric-value">Web + AI</div><div class="metric-note">Independent second perspective</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">SAVED RECORDS</div><div class="metric-value">{saved_count}</div><div class="metric-note">Stored in answers.txt</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Start a support conversation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Describe the issue in your own words. Be specific for the most useful response.</div>',
    unsafe_allow_html=True,
)

suggestions = [
    "How do I reset my password?",
    "Explain a refund or cancellation policy.",
    "Help me troubleshoot a technical issue.",
]
suggestion_cols = st.columns(3, gap="small")
for index, suggestion in enumerate(suggestions):
    with suggestion_cols[index]:
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state["support_query_input"] = suggestion

if "support_query_input" not in st.session_state:
    st.session_state["support_query_input"] = ""

with st.form("support_query_form", border=True):
    query = st.text_area(
        "Your question",
        key="support_query_input",
        placeholder="Tell us what you need help with…",
        height=135,
        help="Avoid including passwords, payment card numbers, or other sensitive information.",
    )
    form_col1, form_col2 = st.columns([3, 1])
    with form_col1:
        st.caption("Your request will be processed by three sequential CrewAI agents.")
    with form_col2:
        submitted = st.form_submit_button(
            "✦  Get support",
            type="primary",
            use_container_width=True,
        )

if submitted:
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        st.session_state.support_result = None
        try:
            with st.status("Working on your request…", expanded=True) as status:
                st.write("Step 1 of 3 · Preparing a direct answer")
                crew, assistant_task, web_task, entry_task = build_support_crew()
                st.write("Step 2 of 3 · Researching trusted web results")
                crew.kickoff(inputs={"query": query.strip()})
                st.write("Step 3 of 3 · Saving the support record")
                assistant_output = (
                    assistant_task.output.raw if assistant_task.output else ""
                )
                web_output = web_task.output.raw if web_task.output else ""
                entry_output = (
                    entry_task.output.raw
                    if entry_task.output
                    else "No Record Keeper output was returned."
                )
                st.session_state.support_result = {
                    "query": query.strip(),
                    "assistant": str(assistant_output),
                    "web": str(web_output),
                    "entry": str(entry_output),
                }
                status.update(
                    label="Support workflow finished",
                    state="complete",
                    expanded=False,
                )
        except Exception:
            logger.exception("Support crew execution failed.")
            st.error(
                "The support workflow could not complete. Check your `.env` "
                "credentials, network connection, installed packages, and the terminal logs."
            )

result = st.session_state.get("support_result")

if result:
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Your support results</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-copy">Question: {result["query"]}</div>',
        unsafe_allow_html=True,
    )

    tab_assistant, tab_web, tab_entry = st.tabs(
        ["✦  Assistant answer", "⌕  Web research", "✓  Save status"]
    )

    with tab_assistant:
        st.markdown('<div class="agent-pill">AGENT 01 · KNOWLEDGE ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown(result["assistant"])
        st.download_button(
            "Download assistant answer",
            data=result["assistant"],
            file_name="assistant_answer.txt",
            mime="text/plain",
            key="download_assistant",
        )

    with tab_web:
        st.markdown('<div class="agent-pill">AGENT 02 · WEB RESEARCH</div>', unsafe_allow_html=True)
        st.markdown(result["web"])
        st.download_button(
            "Download web research",
            data=result["web"],
            file_name="web_research_answer.txt",
            mime="text/plain",
            key="download_web",
        )

    with tab_entry:
        st.markdown('<div class="agent-pill">AGENT 03 · RECORD KEEPER</div>', unsafe_allow_html=True)
        st.markdown(result["entry"])
        if "Successfully saved" in result["entry"] and ANSWERS_FILE.exists():
            st.success(f"Record saved to `{ANSWERS_FILE.name}`.")
        elif "ERROR:" in result["entry"]:
            st.error("The record-keeping tool reported a save error. Check the application logs.")
        elif ANSWERS_FILE.exists():
            st.info(
                "The records file exists. Check the Record Keeper response above "
                "to confirm this specific interaction was saved."
            )
        else:
            st.warning("No records file was found yet.")

        if ANSWERS_FILE.exists():
            try:
                st.download_button(
                    "Download all support records",
                    data=ANSWERS_FILE.read_bytes(),
                    file_name="answers.txt",
                    mime="text/plain",
                    key="download_records",
                )
            except OSError:
                st.warning("The records file could not be read for download.")

st.markdown(
    """
    <div class="footer-note">
      SupportDesk AI · CrewAI orchestration · Answers should be reviewed before being used for critical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
