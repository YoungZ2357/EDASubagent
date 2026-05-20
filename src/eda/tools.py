# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: tools.py
# -------------------------------------------------------------------------

from langchain_core.tools import tool
from config.settings import get_lazy_frame
import polars as pl
import json
from math import isnan, isinf


def _safe_val(v):
    """将 Polars 值转为 JSON 安全的 Python 原生类型。"""
    if v is None:
        return None
    if isinstance(v, float):
        if isnan(v) or isinf(v):
            return None
        return round(v, 6)
    # polars 的 Int / UInt 系列 → Python int
    if hasattr(v, "__int__") and not isinstance(v, (bool, str)):
        return int(v)
    return str(v)


def _dumps(obj: dict | list) -> str:
    return json.dumps(obj, ensure_ascii=False)


@tool
def explore_schema() -> str:
    """获取数据集的完整结构信息：列名称、数据类型、缺失数量、唯一值数量和示例值。

    使用情景：
    - 用户要求获取列名、数据类型或者数据集结构
    - 需要进行统计分析但不确定具体列名或数据类型时，先调用此工具确认

    不要使用此工具进行实际的统计分析或分布计算。"""
    lf = get_lazy_frame()

    schema = lf.collect_schema()
    col_names = schema.names()
    dtypes = [schema[name] for name in col_names]

    total_rows = lf.select(pl.len()).collect().item()

    stats_df = lf.select([
        pl.all().null_count().name.prefix("null_"),
        pl.all().n_unique().name.prefix("unique_"),
    ]).collect()

    head_df = lf.limit(3).collect()

    columns = []
    for name, dtype in zip(col_names, dtypes):
        null_cnt = int(stats_df[0, f"null_{name}"])
        unique_cnt = int(stats_df[0, f"unique_{name}"])
        samples = [_safe_val(v) for v in head_df[name].to_list()]
        columns.append({
            "name": name,
            "dtype": str(dtype),
            "null_count": null_cnt,
            "null_rate": round(null_cnt / total_rows, 4) if total_rows else 0,
            "n_unique": unique_cnt,
            "sample_values": samples,
        })

    return _dumps({
        "total_rows": total_rows,
        "total_columns": len(col_names),
        "columns": columns,
    })


@tool
def get_descriptive_stats(columns: list[str]) -> str:
    """对指定列生成描述性统计。columns 必须是精确列名，且为数值类型。

    使用情景：
    - 用户询问具体统计量时，如均值、中位数、标准差或分位数

    不要使用此工具进行分布分析或相关性分析。"""
    lf = get_lazy_frame()

    if not columns:
        return _dumps({"error": "未指定任何列。若列名未知，请先调用 explore_schema 获取数据结构。"})

    schema = lf.collect_schema()
    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    non_numeric = [c for c in columns if c not in numeric_cols]

    if not numeric_cols:
        return _dumps({
            "error": "指定的列均为非数值类型，无法计算描述性统计。",
            "non_numeric_columns": non_numeric,
        })

    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).count().alias(f"{col}__count"),
            pl.col(col).null_count().alias(f"{col}__null_count"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).median().alias(f"{col}__median"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).quantile(0.25).alias(f"{col}__q1"),
            pl.col(col).quantile(0.75).alias(f"{col}__q3"),
            pl.col(col).max().alias(f"{col}__max"),
        ])

    stats_row = lf.select(agg_exprs).collect()

    results = []
    for col in numeric_cols:
        results.append({
            "column": col,
            "count": _safe_val(stats_row[0, f"{col}__count"]),
            "null_count": _safe_val(stats_row[0, f"{col}__null_count"]),
            "mean": _safe_val(stats_row[0, f"{col}__mean"]),
            "median": _safe_val(stats_row[0, f"{col}__median"]),
            "std": _safe_val(stats_row[0, f"{col}__std"]),
            "min": _safe_val(stats_row[0, f"{col}__min"]),
            "q1": _safe_val(stats_row[0, f"{col}__q1"]),
            "q3": _safe_val(stats_row[0, f"{col}__q3"]),
            "max": _safe_val(stats_row[0, f"{col}__max"]),
        })

    output = {"stats": results}
    if non_numeric:
        output["skipped_non_numeric"] = non_numeric

    return _dumps(output)


@tool
def get_distribution(column: str, bins: int = 10) -> str:
    """分析指定列的分布情况。数值列返回分箱统计，分类列返回频率表。
    column 必须是精确列名。

    使用情景：
    - 用户询问某列的分布、频率、最常出现的值或者取值范围

    不要使用此工具进行描述性统计（均值、标准差等）或相关性分析。"""
    lf = get_lazy_frame()
    schema = lf.collect_schema()
    dtype = schema[column]

    if dtype.is_numeric():
        col_data = lf.select(column).collect().to_series()
        total = len(col_data)
        null_count = col_data.null_count()

        min_val = col_data.min()
        max_val = col_data.max()
        edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]

        cut_result = (
            col_data
            .cut(edges[1:-1])
            .value_counts()
            .sort(column)
        )

        bin_list = []
        for row in cut_result.iter_rows(named=True):
            bin_list.append({
                "bin": str(row[column]),
                "count": int(row["count"]),
            })

        return _dumps({
            "column": column,
            "dtype": "numeric",
            "total": total,
            "null_count": null_count,
            "min": _safe_val(min_val),
            "max": _safe_val(max_val),
            "bins": bin_list,
        })

    else:
        col_series = lf.select(column).collect().to_series()
        total = len(col_series)
        null_count = col_series.null_count()
        full_unique = col_series.n_unique()

        top_n = 20
        freq_df = (
            col_series
            .value_counts()
            .sort("count", descending=True)
            .head(top_n)
        )

        freq_list = []
        for row in freq_df.iter_rows(named=True):
            cnt = int(row["count"])
            freq_list.append({
                "value": _safe_val(row[column]),
                "count": cnt,
                "proportion": round(cnt / total, 4) if total else 0,
            })

        return _dumps({
            "column": column,
            "dtype": "categorical",
            "total": total,
            "null_count": null_count,
            "n_unique": full_unique,
            "truncated": full_unique > top_n,
            "frequencies": freq_list,
        })


@tool
def get_pearson_correlation(columns: list[str]) -> str:
    """计算指定数值列之间的 Pearson 相关系数矩阵。columns 必须是精确列名。

    使用情景：
    - 用户询问列之间的相关性、关联程度或相关系数

    不要使用此工具进行描述性统计或分布分析。"""
    lf = get_lazy_frame()

    if len(columns) < 2:
        return _dumps({"error": "至少需要 2 列才能计算相关性。"})

    schema = lf.collect_schema()
    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    non_numeric = [c for c in columns if c not in numeric_cols]

    if len(numeric_cols) < 2:
        return _dumps({
            "error": "数值列不足 2 列，无法计算相关性。",
            "skipped_non_numeric": non_numeric,
        })

    # 只计算上三角，避免重复
    df = lf.select(numeric_cols).collect()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            val = df.select(pl.corr(a, b)).item()
            pairs.append({
                "column_a": a,
                "column_b": b,
                "correlation": _safe_val(val),
            })

    # 按绝对值降序，LLM 更容易抓到重点
    pairs.sort(key=lambda p: abs(p["correlation"] or 0), reverse=True)

    output = {
        "columns_analyzed": numeric_cols,
        "pairs": pairs,
    }
    if non_numeric:
        output["skipped_non_numeric"] = non_numeric

    return _dumps(output)