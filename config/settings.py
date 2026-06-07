# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: settings.py
# Date: 2026/5/19 11:01
# -------------------------------------------------------------------------
import os
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable

from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from enum import Enum
import polars as pl
load_dotenv()

class DeepSeekModel(str, Enum):
    FLASH = "deepseek-v4-flash"
    PRO = "deepseek-v4-pro"


def get_llm(
    model: DeepSeekModel = DeepSeekModel.FLASH,
    temperature: float = 0.0
) -> ChatDeepSeek:
    assert 0.0 <= temperature <= 1.0, "模型温度应当不低于0，不高于1"
    return ChatDeepSeek(
        model=model.value,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )


def get_tool_llm(
        model: DeepSeekModel = DeepSeekModel.FLASH,
        temperature: float = 0.0,
) -> Runnable[Any, AIMessage]:
    """
    获取已绑定 EDA 工具的 ChatDeepSeek 对象。
    工具列表从 src.eda.tools 延迟导入，避免循环引用。
    """
    from src.eda.tools import (
        explore_schema,
        get_descriptive_stats,
        get_distribution,
        correlation_analysis
    )  # 延迟导入，防止循环依赖
    eda_tools = [explore_schema, get_descriptive_stats, get_distribution, correlation_analysis]
    llm = get_llm(model=model, temperature=temperature)

    return llm.bind_tools(eda_tools)


SUMMARY_TURN_THRESHOLD: int = 4

# HITL 功能开关：True 时 TUI 渲染确认状态条且 graph 配置 interrupt。
# 当前 HITL 尚未实现，保持 False 直到 Module 4 开始。
HITL_ENABLED: bool = False

_lazy_frame: pl.LazyFrame | None = None

def load_dataset(path: str) -> None:
    global _lazy_frame
    _lazy_frame = pl.scan_csv(path)

def get_lazy_frame() -> pl.LazyFrame:

    if _lazy_frame is None:
        raise RuntimeError("Dataset not loaded")
    return _lazy_frame




# _df: pl.DataFrame | None = None
#
# def load_dataset_instant(path: str) -> None:
#     global _df
#     _df = pl.read_csv(path)
#
# def get_dataframe() -> pl.DataFrame:
#     if _df is None:
#         raise RuntimeError("Dataset not loaded")
#     return _df