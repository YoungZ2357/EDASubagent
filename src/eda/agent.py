# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: agent.py
# Date: 2026/5/19 10:43
# -------------------------------------------------------------------------
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from src.eda.state import EDAState
from src.eda.nodes import init_schema, react_node
from src.eda.edges import tools_condition, after_init_condition
from src.eda.tools import (
    explore_schema,
    get_descriptive_stats,
    get_distribution,
    get_pearson_correlation,
)

_tools = [explore_schema, get_descriptive_stats, get_distribution, get_pearson_correlation]

builder = StateGraph(EDAState)

builder.add_node("init_schema", init_schema)
builder.add_node("react_node", react_node)
builder.add_node("tools", ToolNode(_tools))

builder.add_edge(START, "init_schema")
builder.add_conditional_edges("init_schema", after_init_condition)
builder.add_conditional_edges("react_node", tools_condition)
builder.add_edge("tools", "react_node")

graph = builder.compile()
