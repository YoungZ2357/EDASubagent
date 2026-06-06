# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: main.py
# Date: 2026/5/19
# -------------------------------------------------------------------------
import argparse
import os

from langfuse.langchain import CallbackHandler

from src.eda.agent import init_session, ask
from src.eda.schemas import EDAInput

langfuse_handler = CallbackHandler()


def main():
    parser = argparse.ArgumentParser(description="EDA Sub-Agent")
    parser.add_argument("--file", required=True, help="CSV 文件路径")
    args = parser.parse_args()

    config = {"callbacks": [langfuse_handler]}
    state = init_session(EDAInput(file_path=args.file), config=config)
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

        state, output = ask(state, user_input, config=config)
        print(f"\nAssistant: {output.answer}")


if __name__ == "__main__":
    main()
