# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: agent.py
# Date: 2026/5/19 10:43
# -------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.eda.state import EDAState
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
