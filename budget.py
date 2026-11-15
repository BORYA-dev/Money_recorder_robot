"""
Budget Assistant

Flow:
  User input ─┬─ /balance   → show_balance (view current balance)   → end
               ├─ /history   → show_history (view recent transactions) → end
               ├─ /budget N  → set_budget (reset current balance)   → end
               ├─ /alert N   → set_alert (reset low-balance alert threshold) → end
               ├─ /report [YYYY-MM] → monthly_report (spending-by-category breakdown) → end
               └─ anything else → classify (detect expense vs. income, extract amount/category)
                                 → apply (update balance, log transaction, alert if below threshold) → end

Data is stored in budget_state.json, so it persists across restarts. On startup, if a
calendar month has ended since the last time it was reported, that month's report is
printed automatically. Every 2 months, the transaction history is cleared (the balance
itself is kept).

Install:
  pip install langgraph langchain-core langchain-ollama python-dotenv pydantic
  (first install the Ollama app, and run `ollama pull qwen2.5:7b`)
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# ===== Configuration =====
CATEGORIES = ["Transport", "Entertainment", "Groceries", "Housing", "Medical", "Education", "Other"]
CATEGORY_HINTS = {
    "Transport": "taxi, uber, gas/fuel, parking, public transport, flights, train tickets",
    "Entertainment": "shopping, movies, games, streaming subscriptions, concerts, hobbies, bars, drinks",
    "Groceries": "food, dining out, restaurants, supermarket, takeout",
    "Housing": "rent, mortgage, utilities (electricity, water, internet)",
    "Medical": "insurance, health insurance, doctor visits, pharmacy, medicine",
    "Education": "tuition, books, courses, school supplies",
    "Other": "anything that doesn't clearly fit another category",
}
CURRENCY = "€"
DEFAULT_BUDGET = 10000.0
DEFAULT_ALERT_THRESHOLD = 2000.0
STATE_FILE = Path("budget_state.json")

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)


# ===== State persistence (saved to a file so it survives restarts) =====
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "balance": DEFAULT_BUDGET,
        "alert_threshold": DEFAULT_ALERT_THRESHOLD,
        "history": [],
        "last_reported_month": "",
        "last_cleanup_month": "",
    }


def month_index(month: str) -> int:
    """Convert 'YYYY-MM' to an absolute month count, so months can be subtracted."""
    year, mo = map(int, month.split("-"))
    return year * 12 + mo


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ===== 1. State =====
class BotState(TypedDict):
    user_input: str
    reply: str


# ===== Structured output schema =====
class Transaction(BaseModel):
    type: str = Field(description="expense: money spent; income: money received or deposited")
    amount: float = Field(description="amount, a positive number")
    category: str = Field(
        description="expense category, must be one of: "
                    + "; ".join(f"{c} ({CATEGORY_HINTS[c]})" for c in CATEGORIES)
                    + "; leave as an empty string if it's income"
    )
    description: str = Field(description="a short description of this transaction, in English")


class TransactionList(BaseModel):
    transactions: list[Transaction] = Field(
        description="One entry per distinct amount mentioned in the message, each with its own type, "
                    "category and description. For example, 'rent 500, insurance 100' produces two "
                    "entries. Empty list if the message doesn't actually report any money being spent "
                    "or received (e.g. questions, requests for information, or anything else that isn't "
                    "reporting a real transaction)."
    )


def invoke_structured(schema, messages):
    """Retry structured output with a different method on failure; return None if both fail."""
    for method in ("function_calling", "json_schema"):
        try:
            result = llm.with_structured_output(schema, method=method).invoke(messages)
        except Exception:
            continue
        if result is not None:
            return result
    return None


# ===== 2. Routing =====
def route_input(state: BotState) -> str:
    text = state["user_input"].strip()
    if text == "/balance":
        return "show_balance"
    if text == "/history":
        return "show_history"
    if text.startswith("/budget"):
        return "set_budget"
    if text.startswith("/alert"):
        return "set_alert"
    if text.startswith("/report"):
        return "monthly_report"
    if text.startswith("/category"):
        return "category_detail"
    if text.startswith("/undo"):
        return "undo_last"
    if text.startswith("/"):
        return "unknown_command"
    return "classify"


def classify(state: BotState):
    """Determine whether this message reports one or more transactions, and extract each one."""
    result = invoke_structured(TransactionList, [
        SystemMessage(
            "The user is tracking their budget. A single message may report more than one transaction "
            "(e.g. 'rent 500, insurance 100') — extract each one separately, with its own type "
            "(expense/income), amount, and category."
        ),
        HumanMessage(state["user_input"]),
    ])
    txs = [tx for tx in (result.transactions if result else []) if tx.amount > 0]
    if not txs:
        return {"reply": "😅 I couldn't find an actual transaction in that message. If you're asking about "
                          "your spending, try /category name, /report, or /history instead."}

    bstate = load_state()
    lines = []
    for tx in txs:
        if tx.type == "income":
            bstate["balance"] += tx.amount
            bstate["history"].append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "type": "income",
                "amount": tx.amount,
                "category": "",
                "description": tx.description,
            })
            lines.append(f"💰 Received {CURRENCY}{tx.amount:.2f} ({tx.description})")
        else:
            category = tx.category if tx.category in CATEGORIES else "Other"
            bstate["balance"] -= tx.amount
            bstate["history"].append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "type": "expense",
                "amount": tx.amount,
                "category": category,
                "description": tx.description,
            })
            lines.append(f"🧾 Recorded: {category} {CURRENCY}{tx.amount:.2f} ({tx.description})")
    save_state(bstate)

    lines.append(f"Current balance: {CURRENCY}{bstate['balance']:.2f}")
    if bstate["balance"] < bstate["alert_threshold"]:
        lines.append(f"⚠️ Balance has dropped below the alert threshold ({CURRENCY}{bstate['alert_threshold']:.2f}), time to save!")
    return {"reply": "\n".join(lines)}


# ===== 3. Command handlers =====
def unknown_command(state: BotState):
    return {"reply": f"🤷 Unknown command '{state['user_input'].split()[0]}'. Commands: /balance, /history, "
                      "/report [YYYY-MM], /category name [YYYY-MM], /undo [n], /budget amount, /alert amount."}


def show_balance(state: BotState):
    bstate = load_state()
    reply = f"💳 Current balance: {CURRENCY}{bstate['balance']:.2f} (alert threshold: {CURRENCY}{bstate['alert_threshold']:.2f})"
    return {"reply": reply}


def show_history(state: BotState):
    bstate = load_state()
    history = bstate["history"][-10:]
    if not history:
        return {"reply": "No transactions yet. Just type what you spent or received!"}

    lines = [f"📒 Last {len(history)} transactions"]
    for h in reversed(history):
        sign = "+" if h["type"] == "income" else "-"
        tag = h["category"] or "Income"
        lines.append(f"- [{h['time'][:10]}] {tag} {sign}{CURRENCY}{h['amount']:.2f}: {h['description']}")
    return {"reply": "\n".join(lines)}


def build_monthly_report(bstate: dict, month: str) -> str:
    """Build a spending-by-category report for the given month (format: YYYY-MM)."""
    entries = [h for h in bstate["history"] if h["time"].startswith(month)]
    if not entries:
        return f"📊 No transactions recorded for {month}."

    by_category = defaultdict(float)
    total_expense = 0.0
    total_income = 0.0
    for h in entries:
        if h["type"] == "expense":
            by_category[h["category"]] += h["amount"]
            total_expense += h["amount"]
        else:
            total_income += h["amount"]

    lines = [f"📊 Monthly report for {month}"]
    for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
        pct = (amount / total_expense * 100) if total_expense else 0
        lines.append(f"- {category}: {CURRENCY}{amount:.2f} ({pct:.0f}%)")
    lines.append(f"Total expenses: {CURRENCY}{total_expense:.2f}")
    lines.append(f"Total income: {CURRENCY}{total_income:.2f}")
    lines.append(f"Net change: {CURRENCY}{total_income - total_expense:.2f}")
    return "\n".join(lines)


def monthly_report(state: BotState):
    parts = state["user_input"].split()
    month = parts[1] if len(parts) == 2 else datetime.now().strftime("%Y-%m")
    bstate = load_state()
    return {"reply": build_monthly_report(bstate, month)}


def category_detail(state: BotState):
    """List individual transactions for one category, e.g. /category Food [YYYY-MM]."""
    parts = state["user_input"].split(maxsplit=2)
    if len(parts) < 2:
        return {"reply": f"Usage: /category name [YYYY-MM], e.g. /category Food\nCategories: {', '.join(CATEGORIES)}"}

    matched = next((c for c in CATEGORIES if c.lower() == parts[1].lower()), None)
    if matched is None:
        return {"reply": f"Unknown category '{parts[1]}'. Categories: {', '.join(CATEGORIES)}"}

    month = parts[2] if len(parts) == 3 else datetime.now().strftime("%Y-%m")
    bstate = load_state()
    entries = [
        h for h in bstate["history"]
        if h["type"] == "expense" and h["category"] == matched and h["time"].startswith(month)
    ]
    if not entries:
        return {"reply": f"No {matched} expenses recorded for {month}."}

    lines = [f"🔍 {matched} expenses in {month}"]
    total = 0.0
    for h in entries:
        lines.append(f"- [{h['time'][:10]}] {CURRENCY}{h['amount']:.2f}: {h['description']}")
        total += h["amount"]
    lines.append(f"Total: {CURRENCY}{total:.2f}")
    return {"reply": "\n".join(lines)}


def undo_last(state: BotState):
    """Undo the most recent n transactions (default 1), reversing their effect on the balance."""
    parts = state["user_input"].split()
    try:
        n = int(parts[1]) if len(parts) == 2 else 1
    except ValueError:
        return {"reply": "Usage: /undo [n], e.g. /undo or /undo 3"}
    if n <= 0:
        return {"reply": "Usage: /undo [n], e.g. /undo or /undo 3"}

    bstate = load_state()
    if not bstate["history"]:
        return {"reply": "There's nothing to undo."}

    n = min(n, len(bstate["history"]))
    lines = [f"↩️ Undid the last {n} transaction(s):"]
    for _ in range(n):
        last = bstate["history"].pop()
        if last["type"] == "income":
            bstate["balance"] -= last["amount"]
        else:
            bstate["balance"] += last["amount"]
        sign = "+" if last["type"] == "income" else "-"
        lines.append(f"- {sign}{CURRENCY}{last['amount']:.2f} ({last['description']})")
    lines.append(f"Current balance: {CURRENCY}{bstate['balance']:.2f}")
    save_state(bstate)
    return {"reply": "\n".join(lines)}


def set_budget(state: BotState):
    parts = state["user_input"].split()
    if len(parts) != 2:
        return {"reply": "Usage: /budget amount, e.g. /budget 10000"}
    try:
        amount = float(parts[1])
    except ValueError:
        return {"reply": "Amount must be a number, e.g. /budget 10000"}

    bstate = load_state()
    bstate["balance"] = amount
    save_state(bstate)
    return {"reply": f"✅ Balance set to {CURRENCY}{amount:.2f}"}


def set_alert(state: BotState):
    parts = state["user_input"].split()
    if len(parts) != 2:
        return {"reply": "Usage: /alert amount, e.g. /alert 2000"}
    try:
        amount = float(parts[1])
    except ValueError:
        return {"reply": "Amount must be a number, e.g. /alert 2000"}

    bstate = load_state()
    bstate["alert_threshold"] = amount
    save_state(bstate)
    return {"reply": f"✅ Alert threshold set to {CURRENCY}{amount:.2f}"}


# ===== 4. Build the graph =====
graph = StateGraph(BotState)
graph.add_node("classify", classify)
graph.add_node("show_balance", show_balance)
graph.add_node("show_history", show_history)
graph.add_node("set_budget", set_budget)
graph.add_node("set_alert", set_alert)
graph.add_node("monthly_report", monthly_report)
graph.add_node("category_detail", category_detail)
graph.add_node("undo_last", undo_last)
graph.add_node("unknown_command", unknown_command)

graph.add_conditional_edges(START, route_input, {
    "classify": "classify",
    "show_balance": "show_balance",
    "show_history": "show_history",
    "set_budget": "set_budget",
    "set_alert": "set_alert",
    "monthly_report": "monthly_report",
    "category_detail": "category_detail",
    "undo_last": "undo_last",
    "unknown_command": "unknown_command",
})
for node in ["classify", "show_balance", "show_history", "set_budget", "set_alert", "monthly_report", "category_detail", "undo_last", "unknown_command"]:
    graph.add_edge(node, END)

app = graph.compile()


# ===== 5. Interactive loop =====
if __name__ == "__main__":
    print("Let's track your budget! Just type what you spent or received (e.g., lunch 120, received salary 30000).")
    print("Commands: /balance (view balance), /history (view log), /report [YYYY-MM] (monthly report), "
          "/category name [YYYY-MM] (transactions in a category), /undo [n] (undo last n transactions), "
          "/budget amount (reset balance), /alert amount (reset alert threshold), /quit (exit)\n")

    # If a previous month has ended since we last reported on it, show its report automatically.
    bstate = load_state()
    current_month = datetime.now().strftime("%Y-%m")
    past_months = {h["time"][:7] for h in bstate["history"] if h["time"][:7] < current_month}
    last_reported = bstate.get("last_reported_month", "")
    unreported_months = sorted(m for m in past_months if m > last_reported)
    if unreported_months:
        latest = unreported_months[-1]
        print(build_monthly_report(bstate, latest), "\n")
        bstate["last_reported_month"] = latest
        save_state(bstate)

    # Clear expense history every 2 months (balance and alert threshold are kept).
    last_cleanup = bstate.get("last_cleanup_month", "")
    if not last_cleanup:
        bstate["last_cleanup_month"] = current_month
        save_state(bstate)
    elif month_index(current_month) - month_index(last_cleanup) >= 2:
        bstate["history"] = []
        bstate["last_cleanup_month"] = current_month
        save_state(bstate)
        print("🧹 It's been 2 months, so the transaction history has been cleared (balance is unaffected).\n")

    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        result = app.invoke({"user_input": text})
        print(result["reply"], "\n")
