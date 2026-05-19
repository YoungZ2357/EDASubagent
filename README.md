# EDASubagent

A conversational EDA (Exploratory Data Analysis) agent for CSV datasets, built with LangGraph. Ask questions in natural language, get instant statistical insights.

> **Note**: This is a standalone sub-agent designed to be embedded into a larger orchestrator ([KagglerAssistant](docs/KagglerAssistant_Demo_PRD.md)). The focus is on demonstrating LangGraph's core mechanisms—graph construction, state management, and tool-calling loops—rather than exhaustive analysis coverage.

---

## What It Can Do

Load any CSV and ask questions conversationally. The agent picks the right analysis tool automatically.

| Capability | Example Question |
|---|---|
| Dataset overview | "这个数据集有哪些列？有多少缺失值？" |
| Descriptive statistics | "age 和 fare 列的均值、标准差是多少？" |
| Distribution analysis | "Pclass 列的分布是什么？" |
| Correlation matrix | "age、fare 和 survived 之间的相关性如何？" |
| Multi-turn follow-up | "刚才那几列，再看看它们的分布" |

The agent maintains conversation history across turns, so follow-up questions work naturally.

---

## How to Use

### 1. Prerequisites

- Python 3.10+
- A [DeepSeek API key](https://platform.deepseek.com/)

### 2. Install

```bash
git clone <repo-url>
cd EDASubagent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### 3. Configure

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_key_here
```

### 4. Run

```bash
python main.py --file path/to/your/data.csv
```

Example with the included sample data:

```bash
python main.py --file docs/train.csv
```

You'll see a prompt:

```
Dataset loaded: train.csv, input questions to analyze.

你: 
```

Type your question in Chinese or English. Type `exit` to quit.

---

## Architecture

```
START
  │
  ▼
init_schema          ← loads dataset structure into system prompt
  │
  ▼ (only if user message present)
react_node  ◄────────────────────┐
  │                               │
  ├─ tool call requested? ──YES──► tools (execute analysis)
  │
  └─ NO → END
```

**Key design decisions:**

- **LangGraph ReAct loop**: `react_node` ↔ `tools` cycle until the LLM is satisfied, then returns the final answer.
- **Polars LazyFrame**: Dataset is loaded lazily—computation defers until a tool actually needs it, keeping memory usage low for large files.
- **Stateful conversation**: `EDAState` extends LangChain's `MessagesState`, so the full message history is preserved across turns automatically.
- **Tool-bound LLM**: The four analysis tools are bound to the LLM at config time; the model selects and invokes them autonomously.

### Analysis Tools

| Tool | Input | What it returns |
|---|---|---|
| `explore_schema` | *(none)* | Column names, dtypes, null counts, unique counts, sample values |
| `get_descriptive_stats` | `columns: list[str]` | Mean, median, std, Q1, Q3 per numeric column |
| `get_distribution` | `column: str`, `bins: int` | Histogram (numeric) or frequency table (categorical) |
| `get_pearson_correlation` | `columns: list[str]` | Pairwise Pearson correlation matrix |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLM | DeepSeek (deepseek-chat-v4-flash / v4-pro) |
| Data processing | Polars (lazy evaluation) |
| Config | python-dotenv |

---

## Project Structure

```
EDASubagent/
├── main.py               # CLI entry point
├── config/
│   └── settings.py       # LLM factory, dataset loader
├── src/eda/
│   ├── agent.py          # Graph definition and compilation
│   ├── state.py          # EDAState schema
│   ├── nodes.py          # init_schema, react_node
│   ├── edges.py          # Conditional routing logic
│   └── tools.py          # Four EDA analysis tools
├── tui/
│   └── app.py            # Textual TUI (planned, not yet implemented)
└── docs/
    ├── KagglerAssistant_Demo_PRD.md
    └── *.csv             # Sample Kaggle competition data
```

---

## Integration Notes (for KagglerAssistant)

This agent is designed to be called as a sub-agent from an orchestrator. Key integration points:

- **Entry**: Pass an `EDAState` dict with `file_path` set and an initial `HumanMessage` to the compiled graph.
- **Exit**: The graph returns a final `EDAState`; the last `AIMessage` in `state["messages"]` contains the response.
- **Streaming**: The graph supports LangGraph's streaming API (`graph.stream(...)`) for token-level output if the orchestrator needs it.
- **Model swap**: Replace `get_llm()` in `config/settings.py` to use any LangChain-compatible LLM (GPT-4o, Claude, etc.).

---

## Limitations (Demo Scope)

- **No visualization**: Text-only output; charts are out of scope for this sub-agent.
- **CSV only**: No Excel, Parquet, or database support.
- **No test suite**: Unit tests are planned but not implemented.
- **TUI stub**: The Textual UI shell exists but is not functional yet.
