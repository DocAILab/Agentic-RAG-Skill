"""Serve the workflow viewer and read-only artifact APIs on loopback."""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .data import RunStore

_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/xragos-logo.png": ("xragos-logo.png", "image/png"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
_RUN_PATH = re.compile(r"^/api/runs/([A-Za-z0-9_-]{1,128})(/node)?$")
_PRIVATE_KEYS = frozenset({
    "api_key", "apikey", "authorization", "password", "passwd", "secret",
    "access_token", "refresh_token", "credentials", "credential",
})
_TOKEN = re.compile(r"(?<![\w-])sk-[A-Za-z0-9._-]{8,}")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"'<>]+")
_CREDENTIAL_URL = re.compile(r"(https?://)[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|password|access_token|refresh_token|authorization)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def public_value(value: Any) -> Any:
    """Last-line redaction without opening settings or collecting credentials."""
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if str(key).lower().replace("-", "_") in _PRIVATE_KEYS
                       else public_value(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        value = _TOKEN.sub("[REDACTED]", value)
        value = _BEARER.sub("Bearer [REDACTED]", value)
        value = _CREDENTIAL_URL.sub(r"\1[REDACTED]@", value)
        return _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ViewerServer(ThreadingHTTPServer):
    """One local reader; there are no execution or mutation endpoints."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, store: RunStore, *, static_root: Path | None = None):
        self.store = store
        self.store_lock = threading.RLock()
        self.static_root = (static_root or Path(__file__).parent / "static").resolve()
        super().__init__(("127.0.0.1", port), ViewerHandler)


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer
    server_version = "XRAGOSViewer/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Prompts and query parameters never belong in console/access logs.
        return

    def _local_request(self) -> bool:
        expected_port = self.server.server_port
        allowed = {f"127.0.0.1:{expected_port}", f"localhost:{expected_port}"}
        if expected_port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        host = self.headers.get("Host", "").lower()
        if host not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
            except ValueError:
                return False
            if parsed.scheme != "http" or parsed.netloc.lower() not in allowed or parsed.path:
                return False
        return self.headers.get("Sec-Fetch-Site", "") != "cross-site"

    def _respond(self, status: HTTPStatus, body: bytes, content_type: str, *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
        ))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, value: Any, *, head: bool = False) -> None:
        body = json.dumps(public_value(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._respond(status, body, "application/json; charset=utf-8", head=head)

    def do_GET(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        self._get(head=True)

    def _get(self, *, head: bool = False) -> None:
        if not self._local_request():
            self._json(HTTPStatus.FORBIDDEN, {"error": "仅允许从本机同源访问。"}, head=head)
            return
        try:
            if len(self.path) > 8192:
                raise ValueError("请求过长")
            parsed = urlsplit(self.path)
            path = unquote(parsed.path, errors="strict")
            if "\\" in path or "\x00" in path or any(part in {".", ".."} for part in path.split("/")):
                raise ValueError("无效路径")
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "read_only": True}, head=head)
                return
            if path == "/api/runs":
                with self.server.store_lock:
                    result = self.server.store.list_runs()
                self._json(HTTPStatus.OK, result, head=head)
                return
            match = _RUN_PATH.fullmatch(path)
            if match:
                run_id, node_path = match.groups()
                with self.server.store_lock:
                    if node_path:
                        query = parse_qs(parsed.query, max_num_fields=8)
                        accepted = {"task_id", "node_id", "question_id", "call_id", "component"}
                        if set(query) - accepted or any(len(values) != 1 for values in query.values()):
                            raise ValueError("无效查询参数")
                        values = {key: items[0] for key, items in query.items()}
                        if any(len(item) > 512 for item in values.values()):
                            raise ValueError("查询参数过长")
                        if not values.get("task_id") or not values.get("node_id"):
                            raise ValueError("缺少 Task 或流程环节")
                        result = self.server.store.get_node(run_id, **values)
                    else:
                        result = self.server.store.get_run(run_id)
                self._json(HTTPStatus.OK, result, head=head)
                return
            if path in _STATIC and not parsed.query:
                name, mime = _STATIC[path]
                target = (self.server.static_root / name).resolve()
                if not target.is_relative_to(self.server.static_root) or not target.is_file():
                    raise KeyError(path)
                self._respond(HTTPStatus.OK, target.read_bytes(), mime, head=head)
                return
            raise KeyError(path)
        except (KeyError, FileNotFoundError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "未找到运行或流程证据。"}, head=head)
        except (ValueError, UnicodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "请求路径或参数无效。"}, head=head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": "无法读取运行产物。", "error_type": type(exc).__name__}, head=head)

    def _method_not_allowed(self) -> None:
        self.close_connection = True
        self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "此面板只读，不提供执行或写入接口。"})

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="XRAGOS 本地工作流与测试证据可视化（只读）")
    parser.add_argument("--runtime-root", type=Path,
                        default=repo_root.parent / "Preparation" / "runtime",
                        help="包含实验批次或运行目录的父目录；默认同级 Preparation/runtime")
    parser.add_argument("--port", type=int, default=8765, help="本机 HTTP 端口，默认 8765")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("端口必须为 1–65535")
    runtime_root = args.runtime_root.resolve()
    if not runtime_root.is_dir():
        parser.error("runtime-root 必须是现有运行目录")
    store = RunStore(runtime_root=runtime_root, repo_root=repo_root)
    server = ViewerServer(args.port, store)
    print(f"XRAGOS 工作流面板: http://127.0.0.1:{server.server_port}", flush=True)
    print("只读真实运行产物；使用页面刷新按钮读取最新状态。Ctrl+C 退出。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
