"""为 Windows 命令行入口配置可打印多语言数据的 UTF-8 输出。"""

from __future__ import annotations

import sys
from typing import TextIO


def configure_console_utf8() -> None:
    """尽可能把标准输出和错误流切换为 UTF-8，并保留不可配置的测试流。"""
    for stream in (sys.stdout, sys.stderr):
        _reconfigure_stream(stream)


def _reconfigure_stream(stream: TextIO) -> None:
    """对支持 reconfigure 的文本流设置 UTF-8，失败时保持原有流设置。"""
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return
