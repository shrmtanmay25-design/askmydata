import pandas as pd
from agent import DataAgent
from llm import OfflineLLM
df = pd.read_csv("sample_data.csv")
a = DataAgent(OfflineLLM())
for q in ["What is the average customer_rating?",
          "Show total revenue by category as a bar chart",
          "Top 3 orders by revenue",
          "How many orders per region?",
          "Total revenue by region"]:
    r = a.ask(q, df)
    print("\nQ:", q, "| ok:", r["ok"])
    print(r["code"])
    print(r["result"] if not hasattr(r["result"], "to_string") else r["result"].head(8).to_string())
    if r["fig"] is not None: r["fig"].savefig("test_chart.png", dpi=80)
# self-correction test: LLM that fails first, then fixes
class Flaky:
    n = 0
    def complete(self, s, p):
        if "TASK: EXPLAIN" in p: return "ok"
        Flaky.n += 1
        return "```python\nresult = df['revnue'].sum()\n```" if Flaky.n == 1 else "```python\nresult = df['revenue'].sum()\n```"
r = DataAgent(Flaky()).ask("total revenue", df)
print("\nretry test ok:", r["ok"], [t[0] for t in r["trace"]])
# safety test
class Evil:
    def complete(self, s, p): return "```python\nimport os\nresult = os.listdir('.')\n```"
r = DataAgent(Evil()).ask("list files", df)
print("safety test blocked:", not r["ok"], "|", r["error"])
