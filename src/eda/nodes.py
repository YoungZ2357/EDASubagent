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
    QINGYANG_TRIGGER,
    QINGYANG_INTRO_DIRECTIVE,
    select_system_prompt,
)
from config.settings import get_llm, get_tool_llm


def detect_triggers(state: EDAState):
    """彩蛋强制触发：turn 开始处、react_node 之前恰好运行一次。

    判定只看最新一条 HumanMessage（trimming 不会削当前 turn 的它），确定性、扛 trimming：
    - 命中任一触发词 → 置 ``snark_mode = True``，本 turn 即生效（state 更新在边之间提交，
      同一 invoke 内的 react_node 即可读到）；已 latch 则无需重复置位。
    - 命中暗号 ``qingyang`` → 额外置一次性 ``qingyang_intro_pending = True``，
      让本轮回复开场先嘲讽一通（不受 latch 影响，每次念暗号都生效）。
    """
    text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            break  # 只检查最新一条 HumanMessage
    low = text.lower()

    update: dict = {}
    if not state.get("snark_mode") and any(w.lower() in low for w in TRIGGER_WORDS):
        update["snark_mode"] = True
    if QINGYANG_TRIGGER.lower() in low:
        update["qingyang_intro_pending"] = True
    return update


def react_node(state: EDAState):
    llm = get_tool_llm()
    messages = list(state["messages"])
    # 据当前 snark_mode 经集中化选择器重建系统提示（覆盖 _seed_state 注入的种子提示）；
    # schema 沿用现有 explored_schema channel 传入路径。覆盖仅用于本次调用，幂等安全。
    schema = state.get("explored_schema", "")
    system_content = select_system_prompt(state, schema)
    if state.get("qingyang_intro_pending"):
        # 暗号触发那一轮：在系统提示末尾追加一次性「开场嘲讽」指令（仅自然语言，不进工具参数）。
        system_content += QINGYANG_INTRO_DIRECTIVE
    messages[0] = SystemMessage(content=system_content)
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
    # 消费一次性开场嘲讽信号，使其仅作用于 qingyang 触发的那一轮。
    if state.get("qingyang_intro_pending"):
        update["qingyang_intro_pending"] = False
    return update