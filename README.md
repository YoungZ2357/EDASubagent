# EDASubagent

基于 LangGraph 构建的对话式 EDA（探索性数据分析）代理，支持对 CSV 数据集进行自然语言交互式分析。

> **注意**：本项目是 [LangChain Academy「Intro to LangGraph」](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58238107-course-overview) 课程的学习项目。当前代码主要用于实践 LangGraph 的核心机制——图构建、状态管理和工具调用循环——不追求分析功能的完整性。

---

## 功能概览

加载任意 CSV 文件，即可通过自然语言与代理对话。代理会自动选择合适的分析工具执行查询。

| 能力 | 示例问题 |
|---|---|
| 数据集概览 | "这个数据集有哪些列？有多少缺失值？" |
| 描述性统计 | "age 和 fare 列的均值、标准差是多少？" |
| 分布分析 | "Pclass 列的分布是什么？" |
| 相关性矩阵 | "age、fare 和 survived 之间的相关性如何？" |
| 多轮追问 | "刚才那几列，再看看它们的分布" |

代理自动维护对话历史，后续追问可以直接引用前文。

---

## 使用方法

### 1. 环境要求

- Python 3.10+
- [DeepSeek API Key](https://platform.deepseek.com/)

### 2. 安装

```bash
git clone <repo-url>
cd EDASubagent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### 3. 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_key_here
```

### 4. 运行

```bash
python main.py --file path/to/your/data.csv
```

运行后将看到交互提示：

```
Dataset loaded: train.csv, input questions to analyze.

你: 
```

输入中文或英文问题即可。输入 `exit` 退出。

---

## 架构

```
START
  │
  ▼
init_schema          ← 将数据集结构注入 system prompt
  │
  ▼ （仅当存在用户消息时）
react_node  ◄────────────────────┐
  │                               │
  ├─ 需要调用工具？ ──是──► tools（执行分析）
  │
  └─ 否 → END
```

**关键设计决策：**

- **LangGraph ReAct 循环**：`react_node` ↔ `tools` 循环执行，直到 LLM 判断可以给出最终回答。
- **Polars LazyFrame**：数据集采用惰性加载，仅在工具实际需要时才触发计算，对大文件更省内存。
- **有状态对话**：`EDAState` 继承自 LangChain 的 `MessagesState`，自动保存完整消息历史。
- **工具绑定 LLM**：四个分析工具在配置阶段绑定到 LLM，模型自主选择并调用。

### 分析工具

| 工具 | 输入 | 返回值 |
|---|---|---|
| `explore_schema` | *(无)* | 列名、数据类型、空值数量、唯一值数量、示例值 |
| `get_descriptive_stats` | `columns: list[str]` | 每列的均值、中位数、标准差、Q1、Q3 |
| `get_distribution` | `column: str`, `bins: int` | 直方图（数值列）或频率表（分类列） |
| `get_pearson_correlation` | `columns: list[str]` | 两两 Pearson 相关系数矩阵 |

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 代理框架 | LangGraph + LangChain |
| LLM | DeepSeek（deepseek-chat-v4-flash / v4-pro） |
| 数据处理 | Polars（惰性求值） |
| 配置管理 | python-dotenv |

---

## 项目结构

```
EDASubagent/
├── main.py               # CLI 入口
├── config/
│   └── settings.py       # LLM 工厂、数据集加载器
├── src/eda/
│   ├── agent.py          # 图定义与编译
│   ├── state.py          # EDAState 状态定义
│   ├── nodes.py          # init_schema, react_node
│   ├── edges.py          # 条件路由逻辑
│   └── tools.py          # 四个 EDA 分析工具
└── tui/
    └── app.py            # Textual TUI（规划中，尚未实现）
```

---



## 已知局限（Demo 范畴）

- **无可视化**：仅支持文本输出，不包含图表功能。
- **仅限 CSV**：不支持 Excel、Parquet 或数据库等格式。
- **无测试覆盖**：单元测试尚未实现。
- **TUI 未完成**：Textual 界面仅有骨架，尚未可用。
