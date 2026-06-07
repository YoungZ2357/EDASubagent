# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: nodes.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from src.eda.state import EDAState
from src.eda.prompts import (
    HISTORY_SUMMARY_PREFIX,
    SUMMARY_TEMPLATE,
    SUMMARY_PROMPT_INITIAL,
    SUMMARY_PROMPT_MERGE,
    TRIGGER_WORDS,
    OFF_TOPIC_STREAK_THRESHOLD,
    select_system_prompt,
)
from config.settings import get_llm, get_tool_llm


def detect_triggers(state: EDAState):
    """彩蛋强制触发：turn 开始处、react_node 之前恰好运行一次。

    最新一条 HumanMessage 命中触发词即置 ``snark_mode = True``，本 turn 即生效
    （state 更新在边之间提交，同一 invoke 内的 react_node 即可读到）。
    ``snark_mode`` 已 latch 时直接跳过；只看最新 HumanMessage（trimming 不会削当前
    turn 的它），判定确定性、扛 trimming。
    """
    if state.get("snark_mode"):
        return {}
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            low = text.lower()
            if any(w.lower() in low for w in TRIGGER_WORDS):
                return {"snark_mode": True}
            break  # 只检查最新一条 HumanMessage
    return {}


def react_node(state: EDAState):
    llm = get_tool_llm()
    messages = list(state["messages"])
    # 据当前 snark_mode 经集中化选择器重建系统提示（覆盖 _seed_state 注入的种子提示）；
    # schema 沿用现有 explored_schema channel 传入路径。覆盖仅用于本次调用，幂等安全。
    schema = state.get("explored_schema", "")
    messages[0] = SystemMessage(content=select_system_prompt(state, schema))
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
    """每个 turn 的唯一终点：累加轮数、更新跑题计数、必要时回溯触发彩蛋。

    本 turn 是否「跑题」用确定性代理信号判定：从最后一条 HumanMessage 往后扫
    （即当前 turn，尚未被 trimming 削过），本轮零 ToolUse（无任何带 tool_calls 的
    AIMessage）即算 off-topic。连续 off-topic turn 数「超过 6」则置 ``snark_mode``
    （下一 turn 生效）。计数落独立 channel，不被 trimming 削乱；用工具的 turn 视为
    打断连续段、计数归 0。
    """
    messages = state["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn_msgs = messages[last_human:]
    used_tool = any(isinstance(m, AIMessage) and m.tool_calls for m in turn_msgs)
    streak = 0 if used_tool else state.get("off_topic_streak", 0) + 1

    update = {"turn": 1, "off_topic_streak": streak}
    if streak > OFF_TOPIC_STREAK_THRESHOLD and not state.get("snark_mode"):
        update["snark_mode"] = True
    return update