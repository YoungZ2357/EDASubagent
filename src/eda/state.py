# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: state.py
# Date: 2026/5/19 13:36
# -------------------------------------------------------------------------
from typing import Annotated

from langgraph.graph import MessagesState


def _add_turns(current: int, update: int) -> int:
    """``turn`` 字段的 reducer：将各节点返回的增量累加到现有值上。

    ``finish_turn`` 每完成一轮返回 ``{"turn": 1}``，经此 reducer 累计为总轮数。
    """
    return current + update


class EDAState(MessagesState):
    """EDA sub-agent 的内部状态（扁平 TypedDict，继承 ``MessagesState``）。

    - ``messages``：对话历史，由 ``MessagesState`` 提供 add reducer；
      长对话经 ``summarize_conversation`` 节点做 message trimming。
    - ``file_path``：数据集路径，预留字段（当前数据加载经 settings 全局完成）。
    - ``explored_schema``：``init_schema`` 写入的数据集结构快照。
    - ``summary``：历史对话摘要，由 ``summarize_conversation`` 维护、``react_node`` 读取。
    - ``turn``：累计对话轮数，reducer 见 :func:`_add_turns`。
    - ``off_topic_streak``：连续 off-topic（本 turn 零 ToolUse）turn 计数；独立
      channel（缺省 last-write-wins reducer），不靠扫 messages 推算，因此扛得过
      message trimming。由 ``finish_turn`` 每 turn 更新一次。
    - ``snark_mode``：彩蛋「尖酸四川话」人设开关；一旦置 True 即 latch 到会话结束。
      由 ``detect_triggers``（强制触发）/ ``finish_turn``（跑题回溯触发）设置。
    """

    file_path: str
    explored_schema: str
    summary: str
    turn: Annotated[int, _add_turns] = 0
    off_topic_streak: int = 0
    snark_mode: bool = False