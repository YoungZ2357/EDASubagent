# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: state.py
# Date: 2026/5/19 13:36
# -------------------------------------------------------------------------
from langgraph.graph import MessagesState


class EDAState(MessagesState):
    file_path: str
    explored_schema: str