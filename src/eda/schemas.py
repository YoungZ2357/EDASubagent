# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: schemas.py
# -------------------------------------------------------------------------
"""EDA sub-agent 的对外边界契约。

这些 Pydantic 模型定义 sub-agent 的输入/输出契约，与内部的 TypedDict
``EDAState`` 解耦：调用方（main.py，以及后续作品集阶段的 orchestrator）
只依赖这里的契约，不感知 graph 内部的 message 结构。

本模块保持纯数据契约，不引用 ``EDAState`` 或 LangChain message 类型，
因此可脱离 graph 运行时单独校验（见 tests/test_schemas.py）。
"""
from pydantic import BaseModel, Field


class EDAInput(BaseModel):
    """sub-agent 的入口契约。"""

    file_path: str = Field(..., description="待分析 CSV 文件的路径")
    question: str | None = Field(
        default=None, description="可选的首轮提问；为空则仅初始化数据集"
    )


class EDAOutput(BaseModel):
    """sub-agent 单轮交互的输出契约。"""

    answer: str = Field(..., description="本轮 agent 的自然语言回答")
    turn: int = Field(..., description="累计完成的对话轮数")
    summary: str | None = Field(
        default=None, description="当前的历史对话摘要（若已触发摘要）"
    )


__all__ = ["EDAInput", "EDAOutput"]
