# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: edges.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from langgraph.prebuilt import tools_condition

from config.settings import SUMMARY_TURN_THRESHOLD


def entry_condition(state) -> str:
    """图入口路由：依据已持久化的 state 决定从何处开始本次 invoke。

    schema 初始化已外置到 ``init_session``（图外经 ``update_state`` 写入种子 state），
    故入口只负责运行期路由：

    - 已有历史且 ``turn`` 达阈值 → 先 ``summarize_conversation`` 压缩历史。
    - 否则直接进入 ``react_node``。
    """
    if state.get("turn", 0) >= SUMMARY_TURN_THRESHOLD:
        return "summarize_conversation"
    return "react_node"


__all__ = ["tools_condition", "entry_condition"]
