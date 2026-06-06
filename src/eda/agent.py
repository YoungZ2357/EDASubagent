# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: agent.py
# Date: 2026/5/19 10:43
# -------------------------------------------------------------------------
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config.settings import load_dataset
from src.eda.state import EDAState
from src.eda.schemas import EDAInput, EDAOutput
from src.eda.nodes import init_schema, react_node, summarize_conversation, finish_turn
from src.eda.edges import tools_condition, entry_condition, after_init_condition
from src.eda.tools import (
    explore_schema,
    get_descriptive_stats,
    get_distribution,
    correlation_analysis,
)

_tools = [explore_schema, get_descriptive_stats, get_distribution, correlation_analysis]

builder = StateGraph(EDAState)

builder.add_node("init_schema", init_schema)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("react_node", react_node)
builder.add_node("tools", ToolNode(_tools, handle_tool_errors=True))
builder.add_node("finish_turn", finish_turn)

builder.add_conditional_edges(
    START,
    entry_condition,
    {
        "init_schema": "init_schema",
        "summarize_conversation": "summarize_conversation",
        "react_node": "react_node",
    },
)
builder.add_conditional_edges(
    "init_schema",
    after_init_condition,
    {"react_node": "react_node", END: END},
)
builder.add_edge("summarize_conversation", "react_node")
builder.add_conditional_edges(
    "react_node",
    tools_condition,
    {"tools": "tools", END: "finish_turn"},
)
builder.add_edge("tools", "react_node")
builder.add_edge("finish_turn", END)

# 模块级单例 checkpointer：进程内按 thread_id 隔离/持久化各会话记忆，
# 调用方无需再手动持有并回传 EDAState。
_checkpointer = MemorySaver()
graph = builder.compile(checkpointer=_checkpointer)


# --------------------------------------------------------------------------
# 公共接口：封装 EDAInput/EDAOutput ↔ EDAState 的转换，使调用方无需感知
# graph 内部的 message 结构（满足 PRD §8.3 #1）。
# --------------------------------------------------------------------------
def _last_ai_content(state: EDAState) -> str:
    """从 state 中取出最后一条非空 AIMessage 的文本内容。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _with_thread(config: dict | None, thread_id: str) -> dict:
    """把 ``thread_id`` 合并进 invoke 的 config，同时保留调用方传入的其它项
    （如 callbacks）。checkpointer 据此 ``thread_id`` 隔离/恢复各会话记忆。"""
    cfg = dict(config or {})
    cfg["configurable"] = {**cfg.get("configurable", {}), "thread_id": thread_id}
    return cfg


def init_session(inp: EDAInput, config: dict | None = None) -> str:
    """加载数据集并开启一次会话，返回该会话的 ``thread_id``。

    本次 invoke 触发 ``init_schema`` 运行一次（每会话仅此一次），把数据集结构快照
    写入该 thread 的 checkpoint；后续记忆由 :func:`ask` 经 checkpointer 自动延续，
    调用方无需持有 ``EDAState``。

    ``inp.question`` 为预留字段，当前不在初始化时发起首轮提问。
    ``config`` 透传给 ``graph.invoke``（如 callbacks）。
    """
    load_dataset(inp.file_path)
    thread_id = uuid4().hex
    graph.invoke(
        {"messages": [], "file_path": inp.file_path, "explored_schema": ""},
        config=_with_thread(config, thread_id),
    )
    return thread_id


def ask(thread_id: str, question: str, config: dict | None = None) -> EDAOutput:
    """在指定 ``thread_id`` 的会话上追加一轮提问并执行，返回本轮输出契约。

    仅需传入新的提问；其余 state 由 checkpointer 按 ``thread_id`` 自动恢复并续写。
    ``config`` 透传给 ``graph.invoke``（如 callbacks）。
    """
    new_state = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=_with_thread(config, thread_id),
    )
    return EDAOutput(
        answer=_last_ai_content(new_state),
        turn=new_state.get("turn", 0),
        summary=new_state.get("summary") or None,
    )


__all__ = ["graph", "init_session", "ask"]
