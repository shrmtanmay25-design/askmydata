# AskMyData - AI Data Analysis Agent (PS-05)

**Built by:** Nikhil Leelawat - B.Tech AI & Data Science, Arya College of Engineering
**Event:** Internal Hackathon - Agentic AI (6-hour build)

## Problem
Many students, teachers and small businesses have data in CSV files but don't know the pandas steps
needed to analyse it. Asking a plain chatbot is risky because it may guess numbers instead of computing them.

## Solution
AskMyData is a small AI agent. The user uploads a CSV and asks a question in plain language.
The agent writes pandas code, runs it on the real data, fixes its own mistakes, and explains the result.

## How the agent works
1. **Understand** - reads the dataset schema (columns, types, sample rows).
2. **Write code** - the LLM writes pandas/matplotlib code that stores its answer in `result`.
3. **Run in a sandbox** - code is checked (no imports, file access or network) and executed on a copy of the data.
4. **Observe and retry** - if the code errors, the error is sent back to the LLM and it tries again (max 3 tries).
5. **Explain** - the LLM explains the computed result in 2-3 simple sentences. Code and steps are shown to the user.

Short conversation memory lets follow-up questions work ("now only for the West region").

## Components
| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI, CSV upload, charts, agent-trace viewer |
| `agent.py` | Agent loop, safe code execution, retry logic |
| `llm.py` | Gemini free-tier backend + offline rule-based demo backend |
| `sample_data.csv` | Synthetic sales data (made by `make_sample_data.py`) |
| `cli_test.py` | Quick tests: normal questions, self-correction, blocked unsafe code |

## Setup
```bash
pip install -r requirements.txt
# optional, for full natural-language understanding (free key: https://aistudio.google.com)
export GEMINI_API_KEY=your_key        # Windows: set GEMINI_API_KEY=your_key
streamlit run app.py
```
Without a key the app runs in **offline demo mode** (simple questions like totals, averages, top N, group-by, bar charts).
Set `GEMINI_MODEL` if you want a different Gemini model (default `gemini-2.0-flash`).

## Limitations
- Offline mode is rule-based and only handles simple questions.
- The code sandbox blocks common unsafe patterns but is a prototype, not a hardened security boundary.
- The Gemini path needs a valid API key and internet; model names can change over time.
