# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: nodes.py
# Date: 2026/5/19 14:26
# -------------------------------------------------------------------------
from src.eda.state import EDAState
from config.settings import get_tool_llm

def react_node(state: EDAState):
    llm = get_tool_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def init_schema(state: EDAState):
    from src.eda.tools import explore_schema
    from langchain_core.messages import SystemMessage

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