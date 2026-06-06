# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: test_schemas.py
# -------------------------------------------------------------------------
"""边界契约校验：脱离 graph 运行时（PRD §8.3 #2）。"""
import pytest
from pydantic import ValidationError

from src.eda.schemas import EDAInput, EDAOutput


def test_eda_input_minimal():
    inp = EDAInput(file_path="data.csv")
    assert inp.file_path == "data.csv"
    assert inp.question is None


def test_eda_input_with_question():
    inp = EDAInput(file_path="data.csv", question="描述数据集")
    assert inp.question == "描述数据集"


def test_eda_input_requires_file_path():
    with pytest.raises(ValidationError):
        EDAInput()


def test_eda_output_minimal():
    out = EDAOutput(answer="hi", turn=1)
    assert out.answer == "hi"
    assert out.turn == 1
    assert out.summary is None


def test_eda_output_requires_answer_and_turn():
    with pytest.raises(ValidationError):
        EDAOutput(answer="hi")


def test_eda_output_turn_type_validation():
    with pytest.raises(ValidationError):
        EDAOutput(answer="hi", turn="not-an-int")
