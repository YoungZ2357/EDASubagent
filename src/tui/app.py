# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: app.py
# Date: 2026/6/7
# -------------------------------------------------------------------------
"""EDA sub-agent 的 Textual TUI（布局 B：双栏 + Agent 行为栏）。

设计要点（详见 docs/local/TUI.md 与重写计划）：
- **绝不把动态文本插值进 Rich markup 字符串**：所有对话/trace 写入都用 Rich
  ``Text`` 对象（角色标签带 style，正文为纯文本），因此 LLM 输出里出现 ``[``
  等字符也不会触发 MarkupError。
- **流式 token 节流渲染**：worker 线程只往 buffer 累积 token 并标脏，由一个
  ``set_interval`` 定时器整体刷新进行中的回答，避免「每 token 全量重渲染」导致
  的 UI 线程打满 / 卡死。
- 右栏三块（数据概览 / 分析结果 / Agent 行为）固定占比、各自内部滚动、同时常驻。
- 表格由结构化数据**确定性渲染**，不依赖 LLM 逐字打印。
"""
import json
import os
import random
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    DataTable,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
)

from config.settings import HITL_ENABLED
from eda.agent import ask_stream_events, get_explored_schema, init_session
from eda.schemas import EDAInput
from tui.screens import _COMMANDS, SetSecretKeyScreen

_GREETINGS = ["请讲！", "快点把问题端上来罢", "冲刺！冲刺！冲！冲！"]

# ── 节点名称映射（显示用）──────────────────────────────────────────────────
_NODE_LABELS: dict[str, str] = {
    "react_node": "agent",
    "tools": "tools",
    "summarize_conversation": "summarize",
    "finish_turn": "finish",
}

# 无 tool_calls 时各节点的默认描述
_NODE_DESC: dict[str, str] = {
    "react_node": "LLM 决策",
    "tools": "工具执行",
    "summarize_conversation": "压缩对话历史",
    "finish_turn": "回合结束",
}

# finish_turn 是内部计数节点，不显示在 trace 里
_SILENT_NODES: frozenset[str] = frozenset({"finish_turn"})

# 进行中回答的刷新间隔（秒）。节流的核心参数：足够流畅，又不至于每 token 重渲染。
_FLUSH_INTERVAL: float = 0.1

# trace 行的节点名列宽（对齐用）
_LABEL_WIDTH: int = 10


# ── 线程 → 主线程通信消息 ──────────────────────────────────────────────────
class StreamEvent(Message):
    """Worker 线程向 Textual 事件循环推送的通用事件载体。"""

    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event
        super().__init__()


# ── 输入框（带斜杠命令内联补全的导航键拦截）────────────────────────────────
class CommandInput(Input):
    """普通单行输入框，但当上方补全栏可见时拦截 ↑/↓/Tab/Esc。

    补全栏（``#command-completion``）由 EDAApp 维护显隐；本类只负责在它可见时
    把导航键转成对补全列表的操作，避免被 Input 默认行为（或焦点切换）吞掉。
    """

    def _completion(self) -> OptionList | None:
        try:
            comp = self.app.query_one("#command-completion", OptionList)
        except Exception:  # noqa: BLE001 — 补全栏尚未挂载时静默跳过
            return None
        return comp if comp.display else None

    def on_key(self, event: events.Key) -> None:
        comp = self._completion()
        if comp is None:
            return
        if event.key == "down":
            comp.action_cursor_down()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            comp.action_cursor_up()
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            idx = comp.highlighted
            if idx is not None:
                # option 的 id 存的是命令 display（形如 "/set-secret-key"）
                display = comp.get_option_at_index(idx).id
                if display:
                    self.value = f"{display} "
                    self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            comp.display = False
            event.prevent_default()
            event.stop()


# ── 主应用 ─────────────────────────────────────────────────────────────────
class EDAApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "KagglerAssistant"
    BINDINGS = [("ctrl+q", "quit", "退出")]

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        # 注意：不能命名为 `_thread_id` —— 那会覆盖 Textual `App._thread_id`
        # （它存事件循环所在线程的 id，run_worker 据此判断是否需跨线程 marshal）。
        # 一旦被 LangGraph 的会话 UUID 覆盖，后续 run_worker 会误走 call_from_thread
        # 并在主线程自我死锁（提交问题后 TUI 卡死的根因）。
        self._session_id: str | None = None
        self._streaming_buf: str = ""
        self._streaming_dirty: bool = False

    # ── 布局 ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            # 左栏：对话区
            with Vertical(id="chat-col"):
                yield RichLog(id="chat-log", markup=False, highlight=False, wrap=True)
                yield Static("", id="streaming-msg", markup=False)
            # 右栏：三固定面板
            with Vertical(id="right-col"):
                yield Label("数据概览", classes="panel-title")
                yield DataTable(id="data-overview", show_cursor=False)
                yield Label("分析结果", classes="panel-title")
                yield DataTable(id="analysis-results", show_cursor=False)
                yield Label("Agent 行为", classes="panel-title")
                yield RichLog(id="agent-trace", markup=False, highlight=False, wrap=True)
        # HITL 状态条：仅在开关开启时挂载（决策点 1）。开关与 graph 的 interrupt
        # 配置必须同源——当前 graph 未配 interrupt，故默认 False 不挂载。
        if HITL_ENABLED:
            yield Static(
                "⏸ 等待确认：[↵] 满意 · 输入调整... · 直接追问",
                id="hitl-bar",
                markup=False,
            )
        # 底部输入区：补全栏（默认隐藏）置于输入框上方。
        with Vertical(id="input-area"):
            yield OptionList(id="command-completion")
            yield CommandInput(placeholder="> ", id="user-input", disabled=True)

    # ── 生命周期 ───────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        # 节流定时器：进行中回答整体刷新，避免每 token 重渲染。
        self.set_interval(_FLUSH_INTERVAL, self._flush_streaming)
        self.run_worker(self._init_worker, thread=True, name="init")

    # ── Worker 函数（后台线程）─────────────────────────────────────────────
    def _init_worker(self) -> None:
        try:
            thread_id = init_session(EDAInput(file_path=self.file_path))
            schema = get_explored_schema(thread_id)
            self._session_id = thread_id
            self.post_message(StreamEvent({"type": "init_done", "schema": schema}))
        except Exception as exc:  # noqa: BLE001 — 后台线程异常需回送到 UI 显示
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))

    def _stream_worker(self, question: str) -> None:
        try:
            for ev in ask_stream_events(self._session_id, question):
                self.post_message(StreamEvent(ev))
        except Exception as exc:  # noqa: BLE001
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))
        finally:
            self.post_message(StreamEvent({"type": "turn_done"}))

    # ── 事件处理（保持轻量，重活交给节流定时器）──────────────────────────
    def on_stream_event(self, message: StreamEvent) -> None:
        e = message.event
        t = e["type"]

        if t == "init_done":
            self._populate_schema(e["schema"])
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("助手: ", "bold green"), (random.choice(_GREETINGS), ""))
            )
            self._enable_input()

        elif t == "token":
            # 只累积 + 标脏，不在这里更新 widget（由 _flush_streaming 节流刷新）。
            self._streaming_buf += e["content"]
            self._streaming_dirty = True

        elif t == "node_active":
            self._handle_node_active(e["node"])

        elif t == "node_done":
            self._handle_node_done(
                e["node"],
                e.get("tool_calls", []),
                e.get("tool_result"),
            )

        elif t == "turn_done":
            self._commit_streaming()
            self._enable_input()

        elif t == "error":
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("错误: ", "bold red"), (e.get("message", "未知错误"), ""))
            )
            self._commit_streaming()
            self._enable_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        # 斜杠命令：本地分发，不发给 agent。
        if value.startswith("/"):
            event.input.value = ""
            self.query_one("#command-completion", OptionList).display = False
            cmd_id = value[1:].split()[0] if len(value) > 1 else ""
            known = {c[0] for c in _COMMANDS}
            if cmd_id in known:
                self._dispatch_command(cmd_id)
            else:
                self.notify(f"未知命令：/{cmd_id}", severity="warning")
            return

        if not self._session_id:
            return
        event.input.value = ""
        event.input.disabled = True
        self._streaming_buf = ""
        self._streaming_dirty = False
        self.query_one("#chat-log", RichLog).write(
            Text.assemble(("你: ", "bold blue"), (value, ""))
        )
        self.run_worker(
            lambda: self._stream_worker(value), thread=True, name="stream"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        comp = self.query_one("#command-completion", OptionList)
        if event.value.startswith("/") and not event.input.disabled:
            self._update_completion(event.value, comp)
        else:
            comp.display = False

    def _update_completion(self, value: str, comp: OptionList) -> None:
        """按 ``/`` 后的前缀过滤命令，重填补全栏并控制显隐。"""
        from textual.widgets.option_list import Option

        prefix = value[1:].lower()
        matches = [c for c in _COMMANDS if c[0].lower().startswith(prefix)]
        comp.clear_options()
        for cmd_id, display, desc in matches:
            # 用 display 作为 option id，供 Tab 补全回读。
            comp.add_option(Option(f"{display}  {desc}", id=display))
        comp.display = bool(matches)
        if matches:
            comp.highlighted = 0

    def _dispatch_command(self, cmd_id: str) -> None:
        if cmd_id == "set-secret-key":
            self.push_screen(SetSecretKeyScreen(), self._on_secret_key_entered)

    def _on_secret_key_entered(self, key: str | None) -> None:
        if not key:
            return
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
            found = False
            new_lines = []
            for line in lines:
                if line.startswith("DEEPSEEK_API_KEY="):
                    new_lines.append(f"DEEPSEEK_API_KEY={key}\n")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.insert(0, f"DEEPSEEK_API_KEY={key}\n")
            env_path.write_text("".join(new_lines), encoding="utf-8")
        else:
            env_path.write_text(f"DEEPSEEK_API_KEY={key}\n", encoding="utf-8")
        os.environ["DEEPSEEK_API_KEY"] = key
        self.notify("DeepSeek API Key 已保存", severity="information")

    # ── 流式回答（左栏）──────────────────────────────────────────────────
    def _flush_streaming(self) -> None:
        """节流刷新：仅在有新 token 时整体重绘进行中的回答。"""
        if not self._streaming_dirty:
            return
        self._streaming_dirty = False
        body = Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
        body.append(" ▌", style="blink")
        self.query_one("#streaming-msg", Static).update(body)

    def _commit_streaming(self) -> None:
        """把进行中的回答定稿写入 chat-log，并清空进行中显示。"""
        if self._streaming_buf:
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
            )
        self._streaming_buf = ""
        self._streaming_dirty = False
        self.query_one("#streaming-msg", Static).update(Text(""))

    def _enable_input(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()

    # ── Agent 行为 trace（append-only RichLog）───────────────────────────────
    def _handle_node_active(self, node: str) -> None:
        if node in _SILENT_NODES:
            return
        label = _NODE_LABELS.get(node, node)
        self._write_trace("▶", "yellow", label, "生成中…")

    def _handle_node_done(
        self,
        node: str,
        tool_calls: list[dict],
        tool_result: dict | None,
    ) -> None:
        if node not in _SILENT_NODES:
            label = _NODE_LABELS.get(node, node)
            desc = (
                self._fmt_tool_calls(tool_calls)
                if tool_calls
                else _NODE_DESC.get(node, "完成")
            )
            self._write_trace("✓", "green", label, desc)

        if tool_result:
            self._render_tool_result(tool_result)

    def _write_trace(self, icon: str, icon_color: str, label: str, desc: str) -> None:
        line = Text.assemble(
            (f"{icon} ", icon_color),
            (f"{label:<{_LABEL_WIDTH}}", "cyan"),
            (desc, ""),
        )
        self.query_one("#agent-trace", RichLog).write(line)

    def _fmt_tool_calls(self, tool_calls: list[dict]) -> str:
        if not tool_calls:
            return "完成"
        tc = tool_calls[0]
        name = tc.get("name", "")
        args = tc.get("args", {})
        arg_str = ", ".join(f"{k}='{v}'" for k, v in list(args.items())[:2])
        return f"{name}({arg_str})" if arg_str else f"{name}()"

    # ── 数据概览面板（确定性渲染，init 填一次）─────────────────────────────
    def _populate_schema(self, schema_json: str) -> None:
        try:
            schema = json.loads(schema_json)
        except (json.JSONDecodeError, TypeError):
            return

        file_name = os.path.basename(self.file_path)
        rows = schema.get("total_rows", "?")
        cols = schema.get("total_columns", "?")
        self.sub_title = f"{file_name} · {rows} 行 × {cols} 列"

        table = self.query_one("#data-overview", DataTable)
        table.add_columns("列名", "类型", "缺失", "唯一值")
        for col in schema.get("columns", []):
            table.add_row(
                str(col.get("name", "")),
                str(col.get("dtype", "")),
                str(col.get("null_count", "")),
                str(col.get("n_unique", "")),
            )

    # ── 分析结果面板（确定性渲染）──────────────────────────────────────────
    def _render_tool_result(self, result: dict) -> None:
        table = self.query_one("#analysis-results", DataTable)
        table.clear(columns=True)

        if "bins" in result:
            # get_distribution（数值型）
            table.add_columns("区间", "计数")
            for b in result.get("bins", []):
                table.add_row(str(b.get("bin", "")), str(b.get("count", "")))

        elif "frequencies" in result:
            # get_distribution（分类型）
            table.add_columns("值", "计数", "比例")
            for f in result.get("frequencies", []):
                prop = f.get("proportion", 0)
                table.add_row(
                    str(f.get("value", "")),
                    str(f.get("count", "")),
                    f"{prop:.1%}",
                )

        elif "stats" in result:
            # get_descriptive_stats
            table.add_columns("列", "均值", "中位数", "标准差", "最小", "最大")
            for s in result.get("stats", []):
                def _f(k: str) -> str:
                    v = s.get(k)
                    return f"{v:.4f}" if v is not None else "-"

                table.add_row(
                    str(s.get("column", "")),
                    _f("mean"), _f("median"), _f("std"), _f("min"), _f("max"),
                )

        elif "results" in result:
            # correlation_analysis
            table.add_columns("列A", "列B", "方法", "相关系数")
            for method, pairs in result.get("results", {}).items():
                for p in pairs:
                    val = p.get("value")
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    table.add_row(
                        str(p.get("column_a", "")),
                        str(p.get("column_b", "")),
                        method,
                        val_str,
                    )
