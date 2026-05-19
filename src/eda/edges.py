# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: edges.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.prebuilt import tools_condition


def after_init_condition(state) -> str:
    has_human = any(isinstance(m, HumanMessage) for m in state["messages"])
    return "react_node" if has_human else END


__all__ = ["tools_condition", "after_init_condition"]
