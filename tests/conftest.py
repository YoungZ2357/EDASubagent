# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: conftest.py
# -------------------------------------------------------------------------
"""测试共享 fixture：构造一份小型 CSV 并加载为全局数据集。"""
import pytest

from config.settings import load_dataset

# 含数值列（age 整型、score 浮点）、分类列（category、city）与缺失值。
_CSV_CONTENT = (
    "age,score,category,city\n"
    "25,1.5,A,X\n"
    "30,2.5,B,Y\n"
    ",3.5,A,X\n"
    "40,4.5,B,Z\n"
    "35,,A,Y\n"
)


@pytest.fixture
def csv_path(tmp_path):
    """写入临时 CSV 并返回其路径（不触碰全局状态）。"""
    p = tmp_path / "data.csv"
    p.write_text(_CSV_CONTENT, encoding="utf-8")
    return str(p)


@pytest.fixture
def loaded_dataset(csv_path):
    """将临时 CSV 设置为 settings 中的全局 LazyFrame，供工具函数读取。"""
    load_dataset(csv_path)
    return csv_path
