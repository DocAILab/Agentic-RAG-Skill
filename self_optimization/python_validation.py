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


def validate_component_context_contract(content: str, *, label: str) -> None:
    """检查 Component 对注入 context 的使用是否符合稳定运行时接口。"""
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise PythonStaticValidationError(
            f"Cannot inspect Component context calls in '{label}': {exc.msg}"
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

    for node in ast.walk(run_node):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(isinstance(operand, ast.Name) and operand.id == "context" for operand in operands):
            if any(isinstance(operator, (ast.In, ast.NotIn)) for operator in node.ops):
                raise PythonStaticValidationError(
                    f"Component context contract failed for '{label}': context is "
                    "not a mapping or iterable; use getattr(context, name, None) "
                    "to detect an optional runtime method"
                )

    vector_search_names = {"search_vector_index"}
    scalar_names: set[str] = set()
    for node in ast.walk(run_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {
            target.id for target in targets if isinstance(target, ast.Name)
        }
        if _is_context_vector_search_lookup(value):
            vector_search_names.update(names)
        if _is_obvious_scalar_expression(value):
            scalar_names.update(names)

    required_search_keywords = {
        "query_text",
        "document_ids",
        "document_texts",
        "top_k",
        "text_format_version",
    }
    for node in ast.walk(run_node):
        if not isinstance(node, ast.Call):
            continue
        if _is_context_method(node.func, "call_model"):
            keywords = {
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            }
            if (
                len(node.args) != 1
                or any(keyword.arg is None for keyword in node.keywords)
                or not keywords <= {"temperature", "max_tokens"}
            ):
                raise PythonStaticValidationError(
                    f"Component context contract failed for '{label}': "
                    "call_model() requires one prompt positional argument and only "
                    "temperature/max_tokens keyword arguments"
                )
        if _is_context_method(node.func, "embed"):
            if len(node.args) != 1 or node.keywords:
                raise PythonStaticValidationError(
                    f"Component context contract failed for '{label}': "
                    "context.embed() requires exactly one sequence-of-texts argument"
                )
            argument = node.args[0]
            if _is_obvious_scalar_expression(argument) or (
                isinstance(argument, ast.Name) and argument.id in scalar_names
            ):
                raise PythonStaticValidationError(
                    f"Component context contract failed for '{label}': "
                    "context.embed() requires a sequence of texts, not one string"
                )
        if not _is_vector_search_call(node.func, vector_search_names):
            continue
        keywords = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        if node.args or keywords != required_search_keywords:
            raise PythonStaticValidationError(
                f"Component context contract failed for '{label}': "
                "search_vector_index() requires keyword arguments "
                f"{sorted(required_search_keywords)}"
            )


def _is_context_method(node: ast.expr, method: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
        and node.attr == method
    )


def _is_context_vector_search_lookup(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id != "getattr" or len(node.args) < 2:
        return False
    return (
        isinstance(node.args[0], ast.Name)
        and node.args[0].id == "context"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "search_vector_index"
    )


def _is_vector_search_call(node: ast.expr, aliases: set[str]) -> bool:
    return (
        _is_context_method(node, "search_vector_index")
        or isinstance(node, ast.Name)
        and node.id in aliases
    )


def _is_obvious_scalar_expression(node: ast.expr | None) -> bool:
    if isinstance(node, (ast.Constant, ast.JoinedStr, ast.Subscript)):
        return isinstance(node, (ast.JoinedStr, ast.Subscript)) or isinstance(
            node.value, str
        )
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "str":
            return True
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"strip", "lower", "upper", "casefold"}
        )
    return False


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
