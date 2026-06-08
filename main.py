# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: EDASubagent
# FileName: main.py
# Date: 2026/5/19
# -------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eda.cli import main

if __name__ == "__main__":
    main()
