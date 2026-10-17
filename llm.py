"""LLM backends for AskMyData.

GeminiLLM  -> Google Gemini REST API (free tier key from https://aistudio.google.com)
OfflineLLM -> tiny rule-based code writer so the demo works with no API key / no internet
"""
import os, re
import requests


class GeminiLLM:
    name = "Gemini"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY missing")

    def complete(self, system, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        r = requests.post(url, params={"key": self.api_key}, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


AGG_WORDS = [
    (("average", "avg", "mean"), "mean"),
    (("total", "sum"), "sum"),
    (("highest", "maximum", "max", "largest", "best"), "max"),
    (("lowest", "minimum", "min", "smallest", "worst"), "min"),
    (("count", "how many", "number of"), "count"),
]
CHART_WORDS = ("chart", "plot", "graph", "visual", "bar", "line", "trend")


class OfflineLLM:
    """Very small rule-based fallback. Not a real LLM - only for demo mode."""
    name = "Offline demo mode"

    def complete(self, system, prompt):
        if "TASK: EXPLAIN" in prompt:
            return "Computed directly from your dataset using pandas (offline demo mode gives a short, template explanation)."
        cols = re.findall(r"^COL\|(.+?)\|(.+?)$", prompt, flags=re.M)
        question = re.search(r"QUESTION: (.+)", prompt).group(1).lower()
        return "```python\n" + self._write_code(question, cols) + "\n```"

    @staticmethod
    def _mentioned(question, names):
        found = []
        for n in names:
            variants = {n.lower(), n.lower().replace("_", " ")}
            if any(v in question for v in variants):
                found.append(n)
        return found

    def _write_code(self, q, cols):
        names = [c[0] for c in cols]
        numeric = [c[0] for c in cols if re.search(r"int|float", c[1])]
        cats = [c for c in names if c not in numeric]
        mentioned = self._mentioned(q, names)
        m_num = [c for c in mentioned if c in numeric]
        m_cat = [c for c in mentioned if c in cats]
        agg = next((a for words, a in AGG_WORDS if any(w in q for w in words)), None)
        want_chart = any(w in q for w in CHART_WORDS)
        top = re.search(r"top (\d+)", q)

        if m_cat and (m_num or agg == "count") and (agg or want_chart or " by " in q or " per " in q or " each " in q):
            g = m_cat[0]
            if agg == "count" or not m_num:
                expr = f"df.groupby('{g}').size().sort_values(ascending=False).rename('count')"
            else:
                expr = f"df.groupby('{g}')['{m_num[0]}'].{agg or 'sum'}().sort_values(ascending=False)"
            code = f"result = {expr}.round(2)\n"
            if want_chart:
                code += "result.plot(kind='bar', color='#14746F')\nplt.ylabel(result.name or 'value')\nplt.tight_layout()\n"
            return code
        if top and m_num:
            n = int(top.group(1))
            return f"result = df.nlargest({n}, '{m_num[0]}')\n"
        if m_num and agg in ("mean", "sum", "max", "min"):
            return f"result = df['{m_num[0]}'].{agg}()\n"
        if agg == "count" or "rows" in q:
            return "result = len(df)\n"
        if "correlat" in q and len(m_num) >= 2:
            return f"result = df['{m_num[0]}'].corr(df['{m_num[1]}'])\n"
        return "result = df.describe(include='all').T\n"
