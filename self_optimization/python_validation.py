"""使用 Ruff 对优化模型生成的 Python 完整替换执行轻量静态检查。"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path


class PythonStaticValidationError(ValueError):
    """表示候选 Python 文件存在语法或未定义名称错误。"""


_OUTPUT_REQUIRED_KEYS = {
    "RetrievalResult": {"documents"},
    "RerankResult": {"documents"},
    "GenerationResult": {"answer"},
    "RewriteResult": {"rewritten_query"},
    "ClassificationResult": {"route", "reason", "confidence"},
    "CritiqueResult": {"approved", "score", "feedback", "issues"},
}


def validate_agentic_slot_references(
    content: str,
    *,
    allowed_slots: set[str],
    label: str,
) -> None:
    """确保 Agentic workflow 只通过 components 调用清单中声明的抽象槽位。"""
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise PythonStaticValidationError(
            f"Cannot inspect Agentic slots in '{label}': {exc.msg}"
        ) from exc
    invalid = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            not isinstance(owner, ast.Name)
            or owner.id != "components"
            or node.func.attr not in {"call", "call_all", "has"}
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            continue
        slot_name = node.args[0].value
        if slot_name not in allowed_slots:
            invalid.add(slot_name)
    if invalid:
        raise PythonStaticValidationError(
            f"Agentic workflow '{label}' references undeclared component slots "
            f"{sorted(invalid)}. Allowed slots: {sorted(allowed_slots)}"
        )


def validate_component_output_contract(
    content: str,
    *,
    output_types: set[str],
    label: str,
) -> None:
    """检查 Component 入口中的字面量返回值是否保留声明能力的必需字段。"""
    required_keys = set().union(
        *(_OUTPUT_REQUIRED_KEYS.get(output_type, set()) for output_type in output_types)
    )
    if not required_keys:
        return
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise PythonStaticValidationError(
            f"Cannot inspect Component output in '{label}': {exc.msg}"
        ) from exc
    run_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run_node is None:
        return
    literal_returns = []
    for node in ast.walk(run_node):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        literal_returns.append(keys)
    invalid = [keys for keys in literal_returns if not required_keys <= keys]
    if invalid:
        raise PythonStaticValidationError(
            f"Component '{label}' violates output contract {sorted(output_types)}: "
            f"literal run() returns must include {sorted(required_keys)}, got "
            f"{[sorted(keys) for keys in invalid]}"
        )


def validate_python_source(content: str, *, label: str) -> None:
    """把候选源码写入临时文件，并检查语法和未定义名称。"""
    with tempfile.TemporaryDirectory(prefix="ragskill-python-check-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(content, encoding="utf-8", newline="\n")
        _run_ruff(path, label=label)


def validate_python_file(path: str | Path) -> None:
    """检查已写入 Skill 工作副本的 Python 文件，作为提交前最终防线。"""
    resolved = Path(path).resolve()
    _run_ruff(resolved, label=str(resolved))


def _run_ruff(path: Path, *, label: str) -> None:
    """运行与运行时安全直接相关的 Ruff 规则并转换为稳定异常。"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--no-cache",
                "--select",
                "E9,F401,F63,F7,F82",
                "--output-format",
                "concise",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PythonStaticValidationError(
            f"Cannot run Python static validation for '{label}': {exc}"
        ) from exc
    if completed.returncode == 0:
        return
    detail = (completed.stdout or completed.stderr).strip()
    if len(detail) > 4_000:
        detail = detail[:4_000] + "\n...[truncated]"
    raise PythonStaticValidationError(
        f"Python static validation failed for '{label}':\n{detail}"
    )
