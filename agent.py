"""AskMyData - a small data-analysis agent.

Loop:  understand data -> write pandas code -> run it in a sandbox
       -> if it fails, send the error back and retry -> explain the result.
"""
import re, io, contextlib, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a careful data analyst working on a pandas DataFrame named `df`.
Write Python code that answers the user's question using ONLY `df`, `pd`, `np` and `plt`.
Rules:
- Put the final answer in a variable named `result` (a number, Series or DataFrame).
- If the user asks for a chart, draw it with plt (matplotlib) - do not call plt.show().
- Do NOT import anything, read/write files, or use the network.
- Never invent numbers. Everything must be computed from df.
- Reply with one ```python code block and nothing else."""

BANNED = [r"\bimport\b", r"\bopen\s*\(", r"\bexec\b", r"\beval\b", r"__", r"\bos\.", r"\bsys\.",
          r"subprocess", r"\bto_csv\b", r"\bto_excel\b", r"\bread_", r"\bcompile\b", r"\binput\s*\("]

SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["len", "range", "min", "max", "sum", "abs", "round", "sorted", "list",
                           "dict", "set", "tuple", "enumerate", "zip", "float", "int", "str", "bool",
                           "print", "any", "all", "reversed", "isinstance"]}


def schema_text(df: pd.DataFrame) -> str:
    lines = [f"ROWS: {len(df)}"]
    for c in df.columns:
        lines.append(f"COL|{c}|{df[c].dtype}")
    lines.append("SAMPLE:\n" + df.head(5).to_string())
    return "\n".join(lines)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def check_code(code: str):
    for pat in BANNED:
        if re.search(pat, code):
            raise ValueError(f"Blocked: code contains a forbidden pattern ({pat}).")


def run_code(code: str, df: pd.DataFrame):
    """Run generated code in a restricted namespace. Returns (result, figure, stdout)."""
    check_code(code)
    plt.close("all")
    ns = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np, "plt": plt}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
    fig = plt.gcf() if plt.get_fignums() else None
    return ns.get("result"), fig, buf.getvalue()


def result_to_text(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(20).to_string()
    return str(result)


class DataAgent:
    def __init__(self, llm):
        self.llm = llm
        self.history = []  # short conversation memory: (question, answer-text)

    def ask(self, question: str, df: pd.DataFrame) -> dict:
        trace = []
        schema = schema_text(df)
        memory = "\n".join(f"Earlier Q: {q}\nEarlier result: {a[:300]}" for q, a in self.history[-3:])
        trace.append(("Understand", f"Read dataset schema: {len(df)} rows, {len(df.columns)} columns."))

        error, code, result, fig = None, None, None, None
        for attempt in range(1, MAX_RETRIES + 1):
            prompt = f"{schema}\n\n{memory}\n\nQUESTION: {question}"
            if error:
                prompt += f"\n\nYour previous code failed.\nCODE:\n{code}\nERROR: {error}\nFix it."
            try:
                code = extract_code(self.llm.complete(SYSTEM_PROMPT, prompt))
                trace.append((f"Write code (try {attempt})", code))
                result, fig, _ = run_code(code, df)
                if result is None and fig is None:
                    raise ValueError("Code did not set `result`.")
                trace.append(("Run + observe", result_to_text(result)[:500] if result is not None else "chart created"))
                error = None
                break
            except Exception as e:  # self-correction loop
                error = f"{type(e).__name__}: {e}"
                trace.append(("Error -> retry", error))
        if error:
            return {"ok": False, "answer": "I could not compute this reliably.", "code": code,
                    "result": None, "fig": None, "trace": trace, "error": error}

        rtxt = result_to_text(result) if result is not None else "(chart only)"
        try:
            answer = self.llm.complete(
                "You explain data results to a non-programmer in 2-3 short sentences. Use only the numbers shown.",
                f"TASK: EXPLAIN\nQUESTION: {question}\nCODE:\n{code}\nRESULT:\n{rtxt}")
        except Exception as e:
            answer = f"Result computed. (Explanation unavailable: {e})"
        trace.append(("Explain", answer))
        self.history.append((question, rtxt))
        return {"ok": True, "answer": answer, "code": code, "result": result, "fig": fig,
                "trace": trace, "error": None}
