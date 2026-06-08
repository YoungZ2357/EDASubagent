import argparse

from tui.app import EDAApp


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA Sub-Agent")
    parser.add_argument("--file", required=True, help="CSV 文件路径")
    args = parser.parse_args()
    EDAApp(file_path=args.file).run()
