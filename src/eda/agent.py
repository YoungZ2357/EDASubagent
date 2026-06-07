# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: agent.py
# Date: 2026/5/19 10:43
# -------------------------------------------------------------------------
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config.settings import load_dataset
from src.eda.state import EDAState
from src.eda.schemas import EDAInput, EDAOutput
from src.eda.nodes import react_node, summarize_conversation, finish_turn
from src.eda.edges import tools_condition, entry_condition
from src.eda.prompts import DATA_ANALYST_SYSTEM_PROMPT
from src.eda.tools import (
    explore_schema,
    get_descriptive_stats,
    get_distribution,
    correlation_analysis,
)

_tools = [explore_schema, get_descriptive_stats, get_distribution, correlation_analysis]

builder = StateGraph(EDAState)

builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("react_node", react_node)
builder.add_node("tools", ToolNode(_tools, handle_tool_errors=True))
builder.add_node("finish_turn", finish_turn)

builder.add_conditional_edges(
    START,
    entry_condition,
    {
        "summarize_conversation": "summarize_conversation",
        "react_node": "react_node",
    },
)
builder.add_edge("summarize_conversation", "react_node")
builder.add_conditional_edges(
    "react_node",
    tools_condition,
    {"tools": "tools", END: "finish_turn"},
)
builder.add_edge("tools", "react_node")
builder.add_edge("finish_turn", END)

# 模块级单例 checkpointer：进程内按 thread_id 隔离/持久化各会话记忆，
# 调用方无需再手动持有并回传 EDAState。
_checkpointer = MemorySaver()
graph = builder.compile(checkpointer=_checkpointer)


# --------------------------------------------------------------------------
# 公共接口：封装 EDAInput/EDAOutput ↔ EDAState 的转换，使调用方无需感知
# graph 内部的 message 结构（满足 PRD §8.3 #1）。
# --------------------------------------------------------------------------
def _last_ai_content(state: EDAState) -> str:
    """从 state 中取出最后一条非空 AIMessage 的文本内容。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _with_thread(config: dict | None, thread_id: str) -> dict:
    """把 ``thread_id`` 合并进 invoke 的 config，同时保留调用方传入的其它项
    （如 callbacks）。checkpointer 据此 ``thread_id`` 隔离/恢复各会话记忆。"""
    cfg = dict(config or {})
    cfg["configurable"] = {**cfg.get("configurable", {}), "thread_id": thread_id}
    return cfg


def _seed_state(file_path: str) -> dict:
    """在图外完成 schema 探索并构造初始 state（等价于原 ``init_schema`` 节点体）：
    探索数据集结构、注入分析师系统提示，作为该会话的种子写入 checkpoint。"""
    schema_str = explore_schema.invoke({})
    system_prompt = DATA_ANALYST_SYSTEM_PROMPT.format(schema=schema_str)
    return {
        "file_path": file_path,
        "explored_schema": schema_str,
        "messages": [SystemMessage(content=system_prompt)],
    }


def init_session(inp: EDAInput, config: dict | None = None) -> str:
    """加载数据集并开启一次会话，返回该会话的 ``thread_id``。

    schema 初始化在图外完成：经 :func:`_seed_state` 探索数据集结构后，用
    ``graph.update_state`` 把种子 state（系统提示 + 结构快照）直接写入该 thread 的
    checkpoint（每会话仅此一次）；后续记忆由 :func:`ask` 经 checkpointer 自动延续，
    调用方无需持有 ``EDAState``。

    ``inp.question`` 为预留字段，当前不在初始化时发起首轮提问。
    ``config`` 透传给 checkpointer 配置（如 callbacks）。
    """
    load_dataset(inp.file_path)
    thread_id = uuid4().hex
    graph.update_state(
        _with_thread(config, thread_id),
        _seed_state(inp.file_path),
    )
    return thread_id


def ask(thread_id: str, question: str, config: dict | None = None) -> EDAOutput:
    """在指定 ``thread_id`` 的会话上追加一轮提问并执行，返回本轮输出契约。

    仅需传入新的提问；其余 state 由 checkpointer 按 ``thread_id`` 自动恢复并续写。
    ``config`` 透传给 ``graph.invoke``（如 callbacks）。
    """
    new_state = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=_with_thread(config, thread_id),
    )
    return EDAOutput(
        answer=_last_ai_content(new_state),
        turn=new_state.get("turn", 0),
        summary=new_state.get("summary") or None,
    )


def ask_stream(thread_id: str, question: str, config: dict | None = None):
    """流式版本的 ask：逐 token yield react_node 的 LLM 回复内容。"""
    cfg = _with_thread(config, thread_id)
    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=cfg,
        stream_mode="messages",
    ):
        if (
            isinstance(chunk, AIMessageChunk)
            and chunk.content
            and metadata.get("langgraph_node") == "react_node"
        ):
            yield chunk.content


def ask_stream_events(thread_id: str, question: str, config: dict | None = None):
    """TUI 专用流式接口：双模式 streaming，yield 类型化事件 dict。

    事件类型：
      {"type": "node_active", "node": str}
          LLM 节点开始 token 输出（实时，来自 messages 路）。
      {"type": "token", "content": str}
          单个 LLM token（左栏对话区消费）。
      {"type": "node_done", "node": str, "tool_calls": list, "tool_result": dict | None}
          节点执行完毕（来自 updates 路）。
          tool_calls: react_node AIMessage 上的工具调用列表。
          tool_result: tools 节点 ToolMessage 解析后的结构化结果（JSON dict）。
    """
    import json
    from langchain_core.messages import ToolMessage

    cfg = _with_thread(config, thread_id)
    _current_node: str | None = None

    for mode, data in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=cfg,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            chunk, metadata = data
            node = metadata.get("langgraph_node")
            if node and node != _current_node:
                _current_node = node
                yield {"type": "node_active", "node": node}
            # 仅 react_node 的 token 进左栏对话区；summarize_conversation 等
            # 其它 LLM 节点的输出不应泄漏到用户可见的回答里。
            if (
                node == "react_node"
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
            ):
                yield {"type": "token", "content": chunk.content}

        elif mode == "updates":
            _current_node = None
            for node_name, state_update in data.items():
                tool_calls: list[dict] = []
                tool_result: dict | None = None
                for msg in state_update.get("messages", []):
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        tool_calls = [
                            {"name": tc["name"], "args": tc["args"]}
                            for tc in msg.tool_calls
                        ]
                    elif isinstance(msg, ToolMessage):
                        try:
                            tool_result = json.loads(msg.content)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                yield {
                    "type": "node_done",
                    "node": node_name,
                    "tool_calls": tool_calls,
                    "tool_result": tool_result,
                }


def get_explored_schema(thread_id: str) -> str:
    """从 checkpoint 读取 explored_schema channel，返回 JSON 字符串。"""
    cfg = _with_thread(None, thread_id)
    state = graph.get_state(cfg)
    return state.values.get("explored_schema", "{}")


__all__ = [
    "graph",
    "init_session",
    "ask",
    "ask_stream",
    "ask_stream_events",
    "get_explored_schema",
]
