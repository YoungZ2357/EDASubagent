# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: edges.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from config.settings import SUMMARY_TURN_THRESHOLD


def after_init_condition(state) -> str:
    has_human = any(isinstance(m, HumanMessage) for m in state["messages"])
    if not has_human:
        return END
    if state.get("turn", 0) >= SUMMARY_TURN_THRESHOLD:
        return "summarize_conversation"
    return "react_node"


__all__ = ["tools_condition", "after_init_condition"]
