"""Protocol export utilities (JSON schemas and TS types) for JSON-RPC.

This module is a Python rewrite of the export functionality tailored for Jinx.
It focuses on the JSON-RPC envelope types defined in `jinx/micro/net/jsonrpc.py`.

Functions:
- generate_types(out_dir, prettier_path=None)
- generate_ts(out_dir, prettier_path=None, options=None)
- generate_json(out_dir)
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

HEADER = "// GENERATED CODE! DO NOT MODIFY BY HAND!\n\n"


@dataclass
class GenerateTsOptions:
    generate_indices: bool = True
    ensure_headers: bool = True
    run_prettier: bool = True


def generate_types(out_dir: str | os.PathLike[str], prettier_path: str | os.PathLike[str] | None = None) -> None:
    generate_ts(out_dir, prettier_path, GenerateTsOptions())
    generate_json(out_dir)


def generate_ts(
    out_dir: str | os.PathLike[str],
    prettier_path: str | os.PathLike[str] | None = None,
    options: GenerateTsOptions | None = None,
) -> None:
    options = options or GenerateTsOptions()
    out = Path(out_dir)
    v2_out = out / "v2"
    _ensure_dir(out)
    _ensure_dir(v2_out)

    # Minimal TS type exports for JSON-RPC envelope types
    ts_defs: Dict[str, str] = _ts_type_defs()
    for name, content in ts_defs.items():
        _write_ts_file(out / f"{name}.ts", content, ensure_header=options.ensure_headers)

    # index.ts with re-exports
    if options.generate_indices:
        index_path = _generate_index_ts(out)
        if options.ensure_headers:
            _prepend_header_if_missing(index_path)
        # v2 index remains empty unless needed
        _generate_index_ts(v2_out)

    # Optionally run Prettier
    if options.run_prettier and prettier_path:
        ts_files = _collect_ts_files_recursive(out)
        if ts_files:
            try:
                subprocess.run(
                    [str(prettier_path), "--write", "--log-level", "warn", *[str(p) for p in ts_files]],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                # Non-fatal if prettier is unavailable
                pass


def generate_json(out_dir: str | os.PathLike[str]) -> None:
    out = Path(out_dir)
    _ensure_dir(out)
    bundle = _build_schema_bundle()
    target = out / "codex_app_server_protocol.schemas.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)


# ---- internals ----

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_ts_files_recursive(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".ts"):
                files.append(Path(dirpath) / fn)
    return sorted(files)


def _write_ts_file(path: Path, content: str, ensure_header: bool) -> None:
    text = content if content.startswith(HEADER) else (HEADER + content if ensure_header else content)
    path.write_text(text, encoding="utf-8")


def _generate_index_ts(out_dir: Path) -> Path:
    entries: List[str] = []
    stems: List[str] = []
    if out_dir.exists():
        for p in sorted(out_dir.glob("*.ts")):
            stem = p.stem
            if stem == "index":
                continue
            stems.append(stem)
    stems = sorted(set(stems))
    for name in stems:
        entries.append(f"export type {{ {name} }} from \"./{name}\";\n")

    # expose v2 namespace if ts files exist
    v2_dir = out_dir / "v2"
    has_v2_ts = any(v2_dir.glob("*.ts"))
    if has_v2_ts:
        entries.append('export * as v2 from "./v2";\n')

    content = HEADER + "".join(entries)
    index_path = out_dir / "index.ts"
    index_path.write_text(content, encoding="utf-8")
    return index_path


def _prepend_header_if_missing(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return
    if content.startswith(HEADER):
        return
    path.write_text(HEADER + content, encoding="utf-8")


def _build_schema_bundle() -> Dict[str, Any]:
    definitions = _jsonrpc_definitions()
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CodexAppServerProtocol",
        "type": "object",
        "definitions": definitions,
    }


def _jsonrpc_definitions() -> Dict[str, Any]:
    # Minimal JSON schemas aligned with jinx.micro.net.jsonrpc dataclasses
    return {
        "RequestId": {
            "title": "RequestId",
            "anyOf": [{"type": "string"}, {"type": "integer"}],
        },
        "JSONRPCRequest": {
            "title": "JSONRPCRequest",
            "type": "object",
            "properties": {
                "id": {"$ref": "#/definitions/RequestId"},
                "method": {"type": "string"},
                "params": {},
            },
            "required": ["id", "method"],
            "additionalProperties": False,
        },
        "JSONRPCNotification": {
            "title": "JSONRPCNotification",
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "params": {},
            },
            "required": ["method"],
            "additionalProperties": False,
        },
        "JSONRPCResponse": {
            "title": "JSONRPCResponse",
            "type": "object",
            "properties": {
                "id": {"$ref": "#/definitions/RequestId"},
                "result": {},
            },
            "required": ["id", "result"],
            "additionalProperties": False,
        },
        "JSONRPCErrorError": {
            "title": "JSONRPCErrorError",
            "type": "object",
            "properties": {
                "code": {"type": "integer"},
                "message": {"type": "string"},
                "data": {},
            },
            "required": ["code", "message"],
            "additionalProperties": False,
        },
        "JSONRPCError": {
            "title": "JSONRPCError",
            "type": "object",
            "properties": {
                "error": {"$ref": "#/definitions/JSONRPCErrorError"},
                "id": {"$ref": "#/definitions/RequestId"},
            },
            "required": ["error", "id"],
            "additionalProperties": False,
        },
        "JSONRPCMessage": {
            "title": "JSONRPCMessage",
            "oneOf": [
                {"$ref": "#/definitions/JSONRPCRequest"},
                {"$ref": "#/definitions/JSONRPCNotification"},
                {"$ref": "#/definitions/JSONRPCResponse"},
                {"$ref": "#/definitions/JSONRPCError"},
            ],
        },
    }


def _ts_type_defs() -> Dict[str, str]:
    # Minimal TypeScript type definitions to mirror JSON-RPC envelopes
    defs: Dict[str, str] = {}
    defs["RequestId"] = """export type RequestId = string | number;\n"""
    defs["JSONRPCRequest"] = (
        """export type JSONRPCRequest = {\n"""
        "  id: RequestId;\n"
        "  method: string;\n"
        "  params?: unknown;\n"
        "};\n"
    )
    defs["JSONRPCNotification"] = (
        """export type JSONRPCNotification = {\n"""
        "  method: string;\n"
        "  params?: unknown;\n"
        "};\n"
    )
    defs["JSONRPCResponse"] = (
        """export type JSONRPCResponse = {\n"""
        "  id: RequestId;\n"
        "  result: unknown;\n"
        "};\n"
    )
    defs["JSONRPCErrorError"] = (
        """export type JSONRPCErrorError = {\n"""
        "  code: number;\n"
        "  message: string;\n"
        "  data?: unknown;\n"
        "};\n"
    )
    defs["JSONRPCError"] = (
        """export type JSONRPCError = {\n"""
        "  error: JSONRPCErrorError;\n"
        "  id: RequestId;\n"
        "};\n"
    )
    defs["JSONRPCMessage"] = (
        """export type JSONRPCMessage =\n"""
        "  | JSONRPCRequest\n"
        "  | JSONRPCNotification\n"
        "  | JSONRPCResponse\n"
        "  | JSONRPCError;\n"
    )
    return defs
