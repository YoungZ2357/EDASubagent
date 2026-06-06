# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: prompts.py
# -------------------------------------------------------------------------
"""集中管理所有提示词与模板。

节点（nodes.py）不再内联硬编码任何提示词文本，统一从本模块引用并以
``str.format(...)`` 注入运行期变量，便于维护、复用与后续调优。
"""

# init_schema 节点：注入数据集结构后的系统提示词。占位符 {schema}。
DATA_ANALYST_SYSTEM_PROMPT = (
    "你是一个数据分析助手，帮助用户对已加载的数据集进行探索性分析。\n"
    "以下是数据集的结构信息，供你参考：\n\n"
    "{schema}"
)

# react_node 节点：把历史对话摘要作为 SystemMessage 注入。占位符 {summary}。
HISTORY_SUMMARY_PREFIX = "[历史对话摘要]\n{summary}"

# summarize_conversation 节点：摘要的固定字段模板。
SUMMARY_TEMPLATE = (
    "[对话摘要]\n"
    "用户目标：<用户核心分析意图>\n"
    "已完成操作：<已调用的工具及结果要点，按时间顺序>\n"
    "关键发现：<分析中的重要结论>\n"
    "待处理：<未完成的任务或用户最新问题>"
)

# 首次生成摘要的指令。占位符 {template}。
SUMMARY_PROMPT_INITIAL = (
    "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
    "将以上对话总结为摘要，总长度不超过500字：\n{template}"
)

# 已有摘要时，将新增消息合并进摘要的指令。占位符 {summary}、{template}。
SUMMARY_PROMPT_MERGE = (
    "已有摘要：\n{summary}\n\n"
    "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
    "将新增消息合并进摘要，总长度不超过500字：\n{template}"
)

__all__ = [
    "DATA_ANALYST_SYSTEM_PROMPT",
    "HISTORY_SUMMARY_PREFIX",
    "SUMMARY_TEMPLATE",
    "SUMMARY_PROMPT_INITIAL",
    "SUMMARY_PROMPT_MERGE",
]
