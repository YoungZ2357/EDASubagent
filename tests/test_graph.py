# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: test_graph.py
# -------------------------------------------------------------------------
"""Graph 流转 smoke 测试：用桩 LLM 替换真实模型，不触网（PRD §8.2）。"""
import pytest
from langchain_core.messages import AIMessage

import src.eda.nodes as nodes
from src.eda.agent import graph, init_session, ask
from src.eda.schemas import EDAInput


class _StubLLM:
    """桩 LLM：返回不含 tool_call 的固定回答，使 react_node 直接收敛到 finish_turn。"""

    def invoke(self, messages):
        return AIMessage(content="桩回答")


@pytest.fixture(autouse=True)
def _stub_llms(monkeypatch):
    # react_node 在调用时按模块名查找 get_tool_llm，故在 nodes 模块上打桩。
    monkeypatch.setattr(nodes, "get_tool_llm", lambda *a, **k: _StubLLM())
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _StubLLM())


def test_graph_compiles():
    assert graph is not None


def test_init_session_loads_schema(csv_path):
    state = init_session(EDAInput(file_path=csv_path))
    # 初始化只跑 init_schema（无 HumanMessage → 直接 END），写入结构快照。
    assert state["explored_schema"]
    assert state.get("turn", 0) == 0


def test_ask_single_turn(csv_path):
    state = init_session(EDAInput(file_path=csv_path))
    state, output = ask(state, "描述这个数据集")
    assert output.answer == "桩回答"
    assert output.turn == 1
