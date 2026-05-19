# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: tools.py
# Date: 2026/5/19 11:02
# -------------------------------------------------------------------------

from langchain_core.tools import tool  # 工具修饰符
from config.settings import get_lazy_frame
import polars as pl

# 不要对工具函数编写Docstring，其内容将作为提示词的一部分

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

    stats_lf = lf.select([
        pl.all().null_count().name.prefix("null_"),
        pl.all().n_unique().name.prefix("unique_")
    ])

    head_lf = lf.limit(3)

    stats_df = stats_lf.collect()
    head_df = head_lf.collect()

    rows = []
    for name, dtype in zip(col_names, dtypes):
        null_cnt = stats_df[0, f"null_{name}"]
        unique_cnt = stats_df[0, f"unique_{name}"]
        sample_vals = head_df[name].to_list()
        sample_str = ", ".join(
            str(v)[:50] + ("..." if len(str(v)) > 50 else "")
            for v in sample_vals
        )
        rows.append({
            "column": name,
            "dtype": str(dtype),
            "null_count": null_cnt,
            "n_unique": unique_cnt,
            "head(3)": sample_str
        })

    overview = pl.DataFrame(rows)

    with pl.Config(
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=300,
        fmt_str_lengths=60,
        set_tbl_cell_numeric_alignment="RIGHT"
    ):
        return str(overview)

@tool
def get_descriptive_stats(columns: list[str]) -> str:
    """对指定列生成描述性统计。columns 必须是精确列名，且为数值类型。

    使用情景：
    - 用户询问具体统计量时，如均值、中位数、标准差或分位数

    不要使用此工具进行分布分析或相关性分析。"""
    lf = get_lazy_frame()

    if not columns:
        return "未指定任何列。若列名未知，请先调用 explore_schema 获取数据结构。"

    # 过滤出数值列
    schema = lf.collect_schema()
    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    non_numeric = [c for c in columns if c not in numeric_cols]

    if not numeric_cols:
        return f"指定的列均为非数值类型（{', '.join(non_numeric)}），无法计算描述性统计。"

    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).median().alias(f"{col}__median"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).quantile(0.25).alias(f"{col}__q1"),
            pl.col(col).quantile(0.75).alias(f"{col}__q3"),
        ])

    stats_row = lf.select(agg_exprs).collect()

    # 直接构建结果，避免 unpivot/pivot
    rows = []
    for col in numeric_cols:
        rows.append({
            "列名": col,
            "均值": stats_row[0, f"{col}__mean"],
            "中位数": stats_row[0, f"{col}__median"],
            "标准差": stats_row[0, f"{col}__std"],
            "Q1": stats_row[0, f"{col}__q1"],
            "Q3": stats_row[0, f"{col}__q3"],
        })

    result = pl.DataFrame(rows)

    output_parts = []
    with pl.Config(
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=300,
        set_tbl_cell_numeric_alignment="RIGHT",
        float_precision=2,
    ):
        output_parts.append(str(result))

    if non_numeric:
        output_parts.append(f"\n以下列为非数值类型，已跳过：{', '.join(non_numeric)}")

    return "\n".join(output_parts)

@tool
def get_distribution(column: str, bins: int = 10) -> str:
    """分析指定列的分布情况。数值列返回分箱统计，定类列返回频率表。
    column必须是精确列名

    使用情景：
    - 用户询问某列的分布、频率、最常出现的值或者取值范围
    -

    不要使用此工具进行描述性统计（均值、标准差等）或相关性分析。
    """
    lf = get_lazy_frame()
    schema = lf.collect_schema()
    dtype = schema[column]

    if dtype.is_numeric():
        # 数值列：分箱统计
        col_data = lf.select(column).collect().to_series()

        min_val = col_data.min()
        max_val = col_data.max()
        edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]

        result = (
            col_data
            .cut(edges[1:-1])  # 内部边界
            .value_counts()
            .sort(column)
        )

        with pl.Config(tbl_rows=-1, tbl_width_chars=300):
            output = f"列: {column} (数值型)\n"
            output += f"范围: {min_val} ~ {max_val}\n"
            output += f"总数: {len(col_data)}, 缺失: {col_data.null_count()}\n\n"
            output += str(result)
        return output

    else:
        # 分类列：频率表
        result = (
            lf.select(column)
            .collect()
            .to_series()
            .value_counts()
            .sort("count", descending=True)
            .head(20)  # 避免高基数列输出过长
        )

        total = lf.select(pl.len()).collect().item()

        with pl.Config(tbl_rows=-1, tbl_width_chars=300):
            output = f"列: {column} (分类型)\n"
            output += f"总数: {total}, 唯一值: {result.height}"
            if result.height == 20:
                output += "（仅展示前 20）"
            output += "\n\n"
            output += str(result)
        return output

@tool
def get_pearson_correlation(columns: list[str]) -> str:
    """计算指定数值列之间的 Pearson 相关系数矩阵。columns 必须是精确列名。

    使用情景：
    - 用户询问列之间的相关性、关联程度或相关系数

    不要使用此工具进行描述性统计或分布分析。"""
    lf = get_lazy_frame()

    if len(columns) < 2:
        return "至少需要 2 列才能计算相关性。"

    schema = lf.collect_schema()
    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    non_numeric = [c for c in columns if c not in numeric_cols]

    if len(numeric_cols) < 2:
        return f"数值列不足 2 列，无法计算相关性。非数值列已跳过：{', '.join(non_numeric)}"

    df = lf.select(numeric_cols).collect()
    corr = df.select(
        pl.corr(a, b).alias(f"{a}__{b}")
        for a in numeric_cols
        for b in numeric_cols
    )

    # 构建矩阵形式
    rows = []
    for i, a in enumerate(numeric_cols):
        row = {"列名": a}
        for j, b in enumerate(numeric_cols):
            row[b] = corr[0, f"{a}__{b}"]
        rows.append(row)

    result = pl.DataFrame(rows)

    output_parts = []
    with pl.Config(
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=300,
        set_tbl_cell_numeric_alignment="RIGHT",
        float_precision=3,
    ):
        output_parts.append(str(result))

    if non_numeric:
        output_parts.append(f"\n以下列为非数值类型，已跳过：{', '.join(non_numeric)}")

    return "\n".join(output_parts)