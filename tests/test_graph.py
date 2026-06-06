# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: test_graph.py
# -------------------------------------------------------------------------
"""Graph 流转 smoke 测试：用桩 LLM 替换真实模型，不触网（PRD §8.2）。"""
import pytest
from langchain_core.messages import AIMessage, SystemMessage

import src.eda.nodes as nodes
from src.eda.agent import graph, init_session, ask
from src.eda.schemas import EDAInput


def _state_of(thread_id: str) -> dict:
    """读取指定会话在 checkpointer 中的持久化 state。"""
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values


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
    thread_id = init_session(EDAInput(file_path=csv_path))
    # 初始化只跑 init_schema（无 HumanMessage → 直接 END），写入结构快照。
    state = _state_of(thread_id)
    assert state["explored_schema"]
    assert state.get("turn", 0) == 0


def test_ask_single_turn(csv_path):
    thread_id = init_session(EDAInput(file_path=csv_path))
    output = ask(thread_id, "描述这个数据集")
    assert output.answer == "桩回答"
    assert output.turn == 1


def test_init_schema_runs_once(csv_path):
    """回归：init_schema 仅在会话开始运行一次，后续轮次不重复探索/重复注入系统提示。"""
    thread_id = init_session(EDAInput(file_path=csv_path))
    for i in range(3):
        output = ask(thread_id, f"第 {i} 个问题")
        # turn 在 checkpointer 上单调累加，佐证记忆按 thread_id 延续。
        assert output.turn == i + 1

    state = _state_of(thread_id)
    system_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
