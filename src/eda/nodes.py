# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: nodes.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from src.eda.state import EDAState
from config.settings import get_llm, get_tool_llm


def react_node(state: EDAState):
    llm = get_tool_llm()
    messages = list(state["messages"])
    summary = state.get("summary", "")
    if summary:
        summary_msg = SystemMessage(content=f"[历史对话摘要]\n{summary}")
        messages = [messages[0], summary_msg] + messages[1:]
    return {"messages": [llm.invoke(messages)], "turn": 1}


def init_schema(state: EDAState):
    from src.eda.tools import explore_schema

    schema_str = explore_schema.invoke({})
    system_prompt = (
        "你是一个数据分析助手，帮助用户对已加载的数据集进行探索性分析。\n"
        "以下是数据集的结构信息，供你参考：\n\n"
        + schema_str
    )
    return {
        "explored_schema": schema_str,
        "messages": [SystemMessage(content=system_prompt)],
    }


def summarize_conversation(state: EDAState):
    summary = state.get("summary", "")
    if summary:
        prompt = (
            f"已有摘要：{summary}\n\n"
            "请基于新增消息扩展摘要，保持下方模板格式，总长度不超过500字："
        )
    else:
        prompt = (
            "请将以上对话总结为摘要，使用以下模板，总长度不超过500字：\n"
            "[对话摘要]\n用户目标：...\n已完成操作：...\n关键发现：...\n待处理：..."
        )

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