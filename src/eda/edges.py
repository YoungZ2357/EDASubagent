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


def entry_condition(state) -> str:
    """图入口路由：依据已持久化的 state 决定从何处开始本次 invoke。

    - ``explored_schema`` 为空 → 首次开启会话，跑 ``init_schema``（每会话仅此一次）。
    - 已有 schema 且 ``turn`` 达阈值 → 先 ``summarize_conversation`` 压缩历史。
    - 否则直接进入 ``react_node``。
    """
    if not state.get("explored_schema"):
        return "init_schema"
    if state.get("turn", 0) >= SUMMARY_TURN_THRESHOLD:
        return "summarize_conversation"
    return "react_node"


def after_init_condition(state) -> str:
    """``init_schema`` 之后的路由：init 仅在会话开始时运行（``turn`` 恒为 0），
    故只需区分本次是否携带提问。"""
    has_human = any(isinstance(m, HumanMessage) for m in state["messages"])
    return "react_node" if has_human else END


__all__ = ["tools_condition", "entry_condition", "after_init_condition"]
