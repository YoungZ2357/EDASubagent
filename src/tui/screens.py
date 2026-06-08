# -*- coding: utf-8 -*-
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

# 斜杠命令的单一数据源：(cmd_id, display, desc)
# 被输入栏上方的内联补全栏与命令分发逻辑共用（见 app.py）。
_COMMANDS: list[tuple[str, str, str]] = [
    ("set-secret-key", "/set-secret-key", "设置 DeepSeek API Key"),
]


class SetSecretKeyScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "取消")]

    def compose(self) -> ComposeResult:
        with Vertical(id="secret-key-dialog"):
            yield Label("设置 DeepSeek API Key", id="sk-title")
            yield Input(placeholder="sk-...", id="sk-input")
            with Horizontal(id="sk-buttons"):
                yield Button("确认", id="sk-confirm", variant="primary")
                yield Button("取消", id="sk-cancel")

    def on_mount(self) -> None:
        self.query_one("#sk-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sk-confirm":
            key = self.query_one("#sk-input", Input).value.strip()
            self.dismiss(key or None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        key = event.value.strip()
        self.dismiss(key or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
