git --versionimport os
import pandas as pd
import streamlit as st
from agent import DataAgent
from llm import GeminiLLM, OfflineLLM

st.set_page_config(page_title="AskMyData", page_icon="📊", layout="wide")
st.title("📊 AskMyData - AI Data Analysis Agent")
st.caption("Upload a CSV, ask questions in plain English. Every number is computed with pandas on your real data.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    key = st.text_input("Gemini API key (free tier)", type="password",
                        value=os.getenv("GEMINI_API_KEY", ""),
                        help="Get one free at aistudio.google.com. Leave empty for offline demo mode.")
    up = st.file_uploader("Upload CSV", type="csv")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("agent", None)
    st.markdown("**Try asking**\n- Total revenue by category as a bar chart\n- Average customer_rating\n- Top 5 orders by revenue\n- How many orders per region?")

df = pd.read_csv(up) if up else pd.read_csv("sample_data.csv")
st.write(f"**Dataset:** {'uploaded file' if up else 'sample_data.csv (synthetic demo data)'} - {df.shape[0]} rows x {df.shape[1]} columns")
with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- agent (rebuilt if the key changes) ----------
mode = "gemini" if key else "offline"
if st.session_state.get("mode") != mode or "agent" not in st.session_state:
    st.session_state.agent = DataAgent(GeminiLLM(api_key=key) if key else OfflineLLM())
    st.session_state.mode = mode
if mode == "offline":
    st.info("Offline demo mode: simple rule-based questions only. Add a Gemini key in the sidebar for full natural-language understanding.")

st.session_state.setdefault("messages", [])

def render(msg):
    st.markdown(msg["answer"])
    if msg.get("fig") is not None:
        st.pyplot(msg["fig"])
    res = msg.get("result")
    if isinstance(res, (pd.DataFrame, pd.Series)):
        st.dataframe(res, use_container_width=True)
    elif res is not None:
        st.metric("Result", f"{res:,.2f}" if isinstance(res, float) else str(res))
    with st.expander("How the agent got this (steps + code)"):
        for step, detail in msg["trace"]:
            st.markdown(f"**{step}**")
            st.code(detail, language="python" if step.startswith("Write") else None)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) if m["role"] == "user" else render(m["payload"])

if q := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            out = st.session_state.agent.ask(q, df)
        render(out)
    st.session_state.messages.append({"role": "assistant", "payload": out})
