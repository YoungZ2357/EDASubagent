# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: agent.py
# Date: 2026/5/19 10:43
# -------------------------------------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config.settings import load_dataset
from src.eda.state import EDAState
from src.eda.schemas import EDAInput, EDAOutput
from src.eda.nodes import init_schema, react_node, summarize_conversation, finish_turn
from src.eda.edges import tools_condition, after_init_condition
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

builder.add_edge(START, "init_schema")
builder.add_conditional_edges(
    "init_schema",
    after_init_condition,
    {"summarize_conversation": "summarize_conversation", "react_node": "react_node", END: END},
)
builder.add_edge("summarize_conversation", "react_node")
builder.add_conditional_edges(
    "react_node",
    tools_condition,
    {"tools": "tools", END: "finish_turn"},
)
builder.add_edge("tools", "react_node")
builder.add_edge("finish_turn", END)

graph = builder.compile()


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


def init_session(inp: EDAInput, config: dict | None = None) -> EDAState:
    """加载数据集并初始化会话，返回初始 state。

    ``inp.question`` 为预留字段，供需要在初始化时直接发起首轮提问的上层
    调用方使用；命令行交互场景下由后续 :func:`ask` 驱动对话。
    ``config`` 透传给 ``graph.invoke``（如 callbacks），调用方无需感知 graph。
    """
    load_dataset(inp.file_path)
    state = graph.invoke(
        EDAState(messages=[], file_path=inp.file_path, explored_schema=""),
        config=config,
    )
    return state


def ask(
    state: EDAState, question: str, config: dict | None = None
) -> tuple[EDAState, EDAOutput]:
    """在现有 state 上追加一轮提问并执行，返回新 state 与本轮输出契约。

    ``config`` 透传给 ``graph.invoke``（如 callbacks）。
    """
    new_state = graph.invoke(
        {**state, "messages": state["messages"] + [HumanMessage(content=question)]},
        config=config,
    )
    output = EDAOutput(
        answer=_last_ai_content(new_state),
        turn=new_state.get("turn", 0),
        summary=new_state.get("summary") or None,
    )
    return new_state, output


__all__ = ["graph", "init_session", "ask"]
