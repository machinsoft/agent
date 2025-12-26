from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from jinx.micro.common.env import truthy

_ran: set[str] = set()


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _emit(event: str, **fields) -> None:
    try:
        s = str(event or "").strip() or "startup.check"
    except Exception:
        s = "startup.check"

    if fields:
        try:
            parts = []
            for k, v in fields.items():
                try:
                    parts.append(f"{k}={v}")
                except Exception:
                    parts.append(f"{k}=<err>")
            if parts:
                s = s + " | " + ", ".join(parts)
        except Exception:
            pass

    # запись в BLUE_WHISPERS если доступен
    try:
        from jinx.log_paths import BLUE_WHISPERS
        root = _repo_root()
        path = os.path.join(root, BLUE_WHISPERS)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except Exception:
        pass

    try:
        from jinx.micro.runtime.crash_diagnostics import record_operation
        record_operation(str(event or "startup.check"), details=dict(fields or {}), success=True)
    except Exception:
        pass

    try:
        from jinx.micro.common.log import log_info
        log_info(str(event or "startup.check"), **fields)
    except Exception:
        try:
            print(s)
        except Exception:
            pass


def _check_python_version(min_major: int = 3, min_minor: int = 9) -> None:
    major, minor = sys.version_info[:2]
    ok = (major > min_major) or (major == min_major and minor >= min_minor)
    if ok:
        _emit("startup.python_version_ok", version=f"{major}.{minor}")
    else:
        _emit("startup.python_version_too_old", version=f"{major}.{minor}", required=f"{min_major}.{min_minor}")


def _check_write_permissions() -> None:
    try:
        with tempfile.NamedTemporaryFile(delete=True) as _:
            _emit("startup.write_permissions_ok")
    except Exception as exc:
        _emit("startup.write_permissions_failed", err=str(exc))


def _check_critical_modules(mods: list[str]) -> None:
    missing = []
    for m in mods:
        try:
            spec = importlib.util.find_spec(m)
            if spec is None or not getattr(spec, "origin", None):
                missing.append(m)
        except Exception:
            missing.append(m)
    if missing:
        _emit("startup.module_spec_missing", count=len(missing), mods=";".join(missing))
    else:
        _emit("startup.module_spec_ok", count=len(mods))


def _check_import_targets(mods: list[str]) -> None:
    failed = []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            if bool(getattr(mod, "__missing__", False)):
                failed.append(m)
        except Exception:
            failed.append(m)

    if failed:
        _emit("startup.import_failed", count=len(failed), mods=";".join(failed))
    else:
        _emit("startup.import_ok", count=len(mods))


def run_startup_checks(stage: str | None = None) -> None:
    if not truthy("JINX_STARTUP_CHECKS", "1"):
        return

    try:
        st = (stage or "post").strip().lower() or "post"
    except Exception:
        st = "post"

    if st in _ran:
        return
    _ran.add(st)

    root = _repo_root()
    _emit("startup.begin", stage=st, root=root)

    _check_python_version()
    _check_write_permissions()

    critical_specs = [
        "jinx.micro.runtime.api",
        "jinx.micro.runtime.verify_integration",
        "jinx.micro.runtime.patch.write_patch",
        "jinx.micro.exec.executor",
        "jinx.micro.exec.run_exports",
        "jinx.micro.sandbox.service",
        "jinx.sandbox_service",
        "jinx.micro.git.ops",
        "jinx.micro.backend.client",
    ]
    _check_critical_modules(critical_specs)

    if st != "pre" and truthy("JINX_STARTUP_IMPORT_CHECKS", "1"):
        import_targets = [
            "jinx.micro.exec.run_exports",
            "jinx.micro.sandbox.service",
            "jinx.sandbox_service",
            "jinx.micro.exec.executor",
        ]
        _check_import_targets(import_targets)

    _emit("startup.end", stage=st)
