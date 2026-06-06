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
    return {"messages": [llm.invoke(messages)]}


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
    _template = (
        "[对话摘要]\n"
        "用户目标：<用户核心分析意图>\n"
        "已完成操作：<已调用的工具及结果要点，按时间顺序>\n"
        "关键发现：<分析中的重要结论>\n"
        "待处理：<未完成的任务或用户最新问题>"
    )
    if summary:
        prompt = (
            f"已有摘要：\n{summary}\n\n"
            "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
            f"将新增消息合并进摘要，总长度不超过500字：\n{_template}"
        )
    else:
        prompt = (
            "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
            f"将以上对话总结为摘要，总长度不超过500字：\n{_template}"
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


def finish_turn(state: EDAState):
    return {"turn": 1}