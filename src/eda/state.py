# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: state.py
# Date: 2026/5/19 13:36
# -------------------------------------------------------------------------
from typing import Annotated

from langgraph.graph import MessagesState


def _add_turns(current: int, update: int) -> int:
    return current + update


class EDAState(MessagesState):
    file_path: str
    explored_schema: str
    summary: str
    turn: Annotated[int, _add_turns] = 0