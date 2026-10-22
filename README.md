# Multi-Agent Customer Support System

A customer support application built with CrewAI and Streamlit. It uses three sequential AI agents to generate a direct answer, research the web, and save both answers to a text file.

## Features

- Three agents executed using CrewAI's sequential process.
- Direct answer from the Customer Support Assistant.
- Independent web research using SerperDevTool.
- Entry Agent saves the query and both answers to `answers.txt`.
- Streamlit interface for submitting queries and viewing results.
- Timestamped records appended to the output file.
- API credentials loaded from environment variables.
- Error logging and basic input validation.

## Architecture

1. **Customer Support Assistant** — answers using its own knowledge.
2. **Web Search Assistant** — researches the query using Serper.
3. **Customer Support Record Keeper** — writes the original query and both answers to `answers.txt`.

The agents run sequentially. The second task receives the first task's output, and the third task receives both preceding task outputs.

## Technology Stack

- Python 3.12
- CrewAI
- CrewAI Tools
- Streamlit
- OpenAI API
- Serper Search API
- python-dotenv

## Project Structure

```text
multi-agent-customer-support/
├── app.py
├── .env                 # Local secrets; do not commit
├── .gitignore
├── requirements.txt
├── answers.txt          # Generated interaction records
└── README.md
```

## Prerequisites

- Python 3.12 or a compatible Python version
- An OpenAI API key
- A Serper API key
- Internet connectivity

## Configuration

Create a local `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key
```

Replace the placeholders with your own credentials. Never commit `.env` or share its contents.

Optionally, configure the model:

```dotenv
OPENAI_MODEL=gpt-4o-mini
```

The application defaults to `gpt-4o-mini` if `OPENAI_MODEL` is not set. Confirm that the selected model is available to your OpenAI account.

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

From the project root:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

`http://localhost:8501`

Enter a question and click **Run Support Crew**. Wait for all three agents to finish.

## Output

Each successful interaction is intended to append a timestamped record to `answers.txt`, containing:

- Timestamp
- Customer query
- Assistant answer
- Web Search Assistant answer

The application displays both answers and the Entry Agent's save status.

## Validation Checklist

- [x] Python syntax check passed.
- [x] Streamlit application launched successfully.
- [x] Both answers displayed in the UI.
- [x] Interaction saved to `answers.txt`.
- [x] Second interaction appended without overwriting the first.
- [x] `.env` is ignored by Git.
- [ ] Test missing API-key handling.
- [ ] Review the final Git diff and confirm no secrets are staged.
- [ ] Commit and push the project to GitHub.

## Security Notes

- Never hard-code API keys in Python files.
- Never commit `.env` or Streamlit secret files.
- Treat AI-generated answers as potentially inaccurate.
- Verify important claims and web sources.
- Avoid storing sensitive customer information in plain-text logs.

## Known Limitations

- Web search depends on Serper API access and availability.
- AI and search requests may incur usage charges.
- The application currently stores records in a local text file.
- Concurrent executions and production-grade access controls are not implemented.

## Assignment

Social Eagle — Gen AI Architect Program
Weekly Buildathon: Multi-Agent Customer Support (CrewAI + Streamlit)
