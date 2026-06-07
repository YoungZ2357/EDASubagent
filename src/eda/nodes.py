# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: nodes.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from src.eda.state import EDAState
from src.eda.prompts import (
    HISTORY_SUMMARY_PREFIX,
    SUMMARY_TEMPLATE,
    SUMMARY_PROMPT_INITIAL,
    SUMMARY_PROMPT_MERGE,
)
from config.settings import get_llm, get_tool_llm


def react_node(state: EDAState):
    llm = get_tool_llm()
    messages = list(state["messages"])
    summary = state.get("summary", "")
    if summary:
        summary_msg = SystemMessage(content=HISTORY_SUMMARY_PREFIX.format(summary=summary))
        messages = [messages[0], summary_msg] + messages[1:]
    return {"messages": [llm.invoke(messages)]}


def summarize_conversation(state: EDAState):
    summary = state.get("summary", "")
    if summary:
        prompt = SUMMARY_PROMPT_MERGE.format(summary=summary, template=SUMMARY_TEMPLATE)
    else:
        prompt = SUMMARY_PROMPT_INITIAL.format(template=SUMMARY_TEMPLATE)

    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = get_llm().invoke(messages)

    # 保留最近 2 轮：找到倒数第 2 个 HumanMessage 的位置作为截断点
    human_indices = [i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)]
    if len(human_indices) > 2:
        cutoff = human_indices[-2]
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:cutoff]]
    else:
        delete_messages = []

    return {"summary": response.content, "messages": delete_messages}


def finish_turn(state: EDAState):
    return {"turn": 1}