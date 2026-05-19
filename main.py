# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: main.py
# Date: 2026/5/19
# -------------------------------------------------------------------------
import argparse
import os
from langchain_core.messages import HumanMessage, AIMessage
from config.settings import load_dataset
from src.eda.agent import graph
from src.eda.state import EDAState


def main():
    parser = argparse.ArgumentParser(description="EDA Sub-Agent")
    parser.add_argument("--file", required=True, help="CSV 文件路径")
    args = parser.parse_args()

    load_dataset(args.file)

    state = graph.invoke(EDAState(
        messages=[],
        file_path=args.file,
        explored_schema="",
    ))
    print(f"数据集已加载：{os.path.basename(args.file)}，输入问题开始分析。")

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        state = graph.invoke({
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)],
        })
        _print_last_ai(state)


def _print_last_ai(state: dict) -> None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\nAssistant: {msg.content}")
            return


if __name__ == "__main__":
    main()
