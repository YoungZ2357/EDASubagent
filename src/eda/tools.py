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
import math

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


def _pearson(df: pl.DataFrame, col_a: str, col_b: str) -> float | None:
    """连续 vs 连续：Pearson 相关系数。"""
    val = df.select(pl.corr(col_a, col_b)).item()
    return val


def _cramers_v(df: pl.DataFrame, col_a: str, col_b: str) -> float | None:
    """分类 vs 分类：Cramér's V。"""
    pair_df = df.select([col_a, col_b]).drop_nulls()
    n = pair_df.height
    if n == 0:
        return None

    # 列联表
    ct = pair_df.group_by([col_a, col_b]).agg(pl.len().alias("observed"))
    row_totals = ct.group_by(col_a).agg(pl.col("observed").sum().alias("row_total"))
    col_totals = ct.group_by(col_b).agg(pl.col("observed").sum().alias("col_total"))

    ct = ct.join(row_totals, on=col_a).join(col_totals, on=col_b)
    ct = ct.with_columns(
        (pl.col("row_total") * pl.col("col_total") / n).alias("expected")
    )

    chi2 = ct.select(
        ((pl.col("observed") - pl.col("expected")).pow(2) / pl.col("expected")).sum()
    ).item()

    r = pair_df.select(pl.col(col_a).n_unique()).item()
    c = pair_df.select(pl.col(col_b).n_unique()).item()
    min_dim = min(r, c) - 1

    if min_dim == 0:
        return None

    return math.sqrt(chi2 / (n * min_dim))


def _eta_squared(df: pl.DataFrame, cat_col: str, num_col: str) -> float | None:
    """分类 vs 连续：Eta²（相关比）。"""
    pair_df = df.select([cat_col, num_col]).drop_nulls()
    if pair_df.height == 0:
        return None

    overall_mean = pair_df.select(pl.col(num_col).mean()).item()
    if overall_mean is None:
        return None

    ss_total = pair_df.select(
        ((pl.col(num_col) - overall_mean).pow(2)).sum()
    ).item()

    if ss_total == 0:
        return None

    groups = pair_df.group_by(cat_col).agg([
        pl.col(num_col).mean().alias("group_mean"),
        pl.len().alias("group_n"),
    ])

    ss_between = groups.select(
        (pl.col("group_n") * (pl.col("group_mean") - overall_mean).pow(2)).sum()
    ).item()

    return ss_between / ss_total




@tool
def correlation_analysis(columns: list[str]) -> str:
    """分析指定列之间的相关性。自动根据列类型选择统计方法：
    - 连续 vs 连续：Pearson 相关系数（-1 到 1）
    - 分类 vs 分类：Cramér's V（0 到 1）
    - 分类 vs 连续：Eta²（0 到 1）

    columns 必须是精确列名，至少 2 列。

    使用情景：
    - 用户询问列之间的相关性、关联程度或相关系数
    - 用户想了解两个或多个变量之间的关系

    不要使用此工具进行描述性统计或分布分析。"""
    lf = get_lazy_frame()

    if len(columns) < 2:
        return _dumps({"error": "至少需要 2 列才能计算相关性。"})

    schema = lf.collect_schema()
    unknown = [c for c in columns if c not in schema.names()]
    if unknown:
        return _dumps({
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        })

    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    categorical_cols = [c for c in columns if not schema[c].is_numeric()]

    df = lf.select(columns).collect()

    results: dict[str, list] = {}

    # 连续 vs 连续 → Pearson
    pearson_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            val = _pearson(df, a, b)
            pearson_pairs.append({
                "column_a": a, "column_b": b, "value": _safe_val(val),
            })
    if pearson_pairs:
        pearson_pairs.sort(key=lambda p: abs(p["value"] or 0), reverse=True)
        results["pearson"] = pearson_pairs

    # 分类 vs 分类 → Cramér's V
    cramers_pairs = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            a, b = categorical_cols[i], categorical_cols[j]
            val = _cramers_v(df, a, b)
            cramers_pairs.append({
                "column_a": a, "column_b": b, "value": _safe_val(val),
            })
    if cramers_pairs:
        cramers_pairs.sort(key=lambda p: abs(p["value"] or 0), reverse=True)
        results["cramers_v"] = cramers_pairs

    # 分类 vs 连续 → Eta²
    eta_pairs = []
    for cat in categorical_cols:
        for num in numeric_cols:
            val = _eta_squared(df, cat, num)
            eta_pairs.append({
                "column_a": cat, "column_b": num, "value": _safe_val(val),
            })
    if eta_pairs:
        eta_pairs.sort(key=lambda p: abs(p["value"] or 0), reverse=True)
        results["eta_squared"] = eta_pairs

    if not results:
        return _dumps({"error": "给定列的组合无法计算任何相关性。"})

    return _dumps({
        "column_types": {
            c: "numeric" if c in numeric_cols else "categorical"
            for c in columns
        },
        "method_descriptions": {
            "pearson": "Pearson 相关系数，范围 -1 到 1，衡量线性相关强度",
            "cramers_v": "Cramér's V，范围 0 到 1，衡量分类变量间的关联强度",
            "eta_squared": "Eta²，范围 0 到 1，衡量分类变量对连续变量的解释力",
        },
        "results": results,
    })



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


