# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: test_tools.py
# -------------------------------------------------------------------------
"""工具函数单测：直接调用 @tool，脱离 graph 运行时（PRD §8.3 #4）。"""
import json

from eda.tools import (
    explore_schema,
    get_descriptive_stats,
    get_distribution,
    correlation_analysis,
)


def _invoke(tool, **kwargs):
    """调用 @tool 并解析其 JSON 字符串输出。"""
    return json.loads(tool.invoke(kwargs))


def test_explore_schema(loaded_dataset):
    out = _invoke(explore_schema)
    assert out["total_rows"] == 5
    assert out["total_columns"] == 4
    by_name = {c["name"]: c for c in out["columns"]}
    assert set(by_name) == {"age", "score", "category", "city"}
    assert by_name["age"]["null_count"] == 1
    assert by_name["score"]["null_count"] == 1
    assert by_name["category"]["n_unique"] == 2


def test_descriptive_stats(loaded_dataset):
    out = _invoke(get_descriptive_stats, columns=["age", "score"])
    stats = {s["column"]: s for s in out["stats"]}
    assert set(stats) == {"age", "score"}
    # age 非空 4 个：25,30,40,35 → min 25 max 40
    assert stats["age"]["count"] == 4
    assert stats["age"]["min"] == 25
    assert stats["age"]["max"] == 40
    assert stats["age"]["null_count"] == 1


def test_descriptive_stats_skips_non_numeric(loaded_dataset):
    out = _invoke(get_descriptive_stats, columns=["age", "category"])
    assert [s["column"] for s in out["stats"]] == ["age"]
    assert out["skipped_non_numeric"] == ["category"]


def test_descriptive_stats_all_non_numeric_errors(loaded_dataset):
    out = _invoke(get_descriptive_stats, columns=["category", "city"])
    assert "error" in out
    assert out["non_numeric_columns"] == ["category", "city"]


def test_descriptive_stats_empty_errors(loaded_dataset):
    out = _invoke(get_descriptive_stats, columns=[])
    assert "error" in out


def test_distribution_numeric(loaded_dataset):
    out = _invoke(get_distribution, column="age", bins=4)
    assert out["dtype"] == "numeric"
    assert out["min"] == 25
    assert out["max"] == 40
    assert out["null_count"] == 1
    assert isinstance(out["bins"], list) and out["bins"]


def test_distribution_categorical(loaded_dataset):
    out = _invoke(get_distribution, column="category")
    assert out["dtype"] == "categorical"
    assert out["n_unique"] == 2
    freqs = {f["value"]: f["count"] for f in out["frequencies"]}
    assert freqs["A"] == 3
    assert freqs["B"] == 2


def test_correlation_analysis(loaded_dataset):
    out = _invoke(correlation_analysis, columns=["age", "score", "category", "city"])
    assert out["column_types"]["age"] == "numeric"
    assert out["column_types"]["category"] == "categorical"
    results = out["results"]
    # 连续×连续、分类×分类、分类×连续 三类方法均应产生条目
    assert "pearson" in results
    assert "cramers_v" in results
    assert "eta_squared" in results


def test_correlation_requires_two_columns(loaded_dataset):
    out = _invoke(correlation_analysis, columns=["age"])
    assert "error" in out


def test_correlation_unknown_column(loaded_dataset):
    out = _invoke(correlation_analysis, columns=["age", "nope"])
    assert "error" in out
    assert "nope" in str(out["error"])
