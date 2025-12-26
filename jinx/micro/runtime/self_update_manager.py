from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.common.log import log_info, log_warn
from jinx.micro.rt.activity import (
    set_activity as _set_act,
    set_activity_detail as _set_det,
    clear_activity as _clear_act,
    clear_activity_detail as _clear_det,
)
from jinx.micro.runtime.self_update_journal import append as _jr
from jinx.micro.runtime.self_update_venv import create_venv, install_requirements
from jinx.micro.runtime.self_update_sign import verify_stage_signature
from jinx.micro.runtime.plugins import publish_event as _pub_evt


_SELFUPDATE_TIMEOUT_S = 18.0
_SELFUPDATE_USE_VENV = True
_SELFUPDATE_GREEN_LOWPRIO = True
_SELFUPDATE_CANARY = True
_SELFUPDATE_STABILIZE_SEC = 10.0
_SELFUPDATE_AUTO_ROLLBACK = True
_SELFUPDATE_DRAIN_SEC = 6.0


class SelfUpdateManager(MicroProgram):
    """Blue-Green self-update orchestrator with preflight and rollback.

    Pipeline (no user interaction required):
    - Stage new code into .jinx/stage/<ts>/
    - Preflight in isolated Python with JINX_PACKAGE_DIR pointing to staged package
    - If preflight OK, spawn green process with handshake file
    - Wait for green to report healthy; on success, gracefully shutdown blue
    - On failure or timeout, rollback (kill green, keep blue), schedule repairs
    """

    def __init__(self) -> None:
        super().__init__(name="SelfUpdateManager")

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("self-update manager online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        name = str(payload.get("name") or "")
        tid = str(payload.get("id") or "")
        if not tid:
            return
        if name != "selfupdate.apply":
            return
        kw = payload.get("kwargs") or {}
        source_dir = (kw.get("source_dir") or "").strip()
        try:
            timeout_s = float(kw.get("timeout_s") or _SELFUPDATE_TIMEOUT_S)
        except Exception:
            timeout_s = _SELFUPDATE_TIMEOUT_S
        await self._apply_self_update(source_dir or None, timeout_s=timeout_s)

    async def _apply_self_update(self, source_dir: Optional[str], *, timeout_s: float) -> None:
        _set_act("self-update")
        _set_det({"update": {"phase": "stage"}})
        _jr("selfupdate.start", stage="stage", ok=None, source=bool(source_dir))
        try:
            stage_dir = await self._stage_code(source_dir)
            if not stage_dir:
                log_warn("selfupdate.stage_failed")
                _jr("selfupdate.stage_failed", stage="stage", ok=False)
                return
            _jr("selfupdate.staged", stage="stage", ok=True, dir=stage_dir)
            _set_det({"update": {"phase": "preflight", "dir": stage_dir}})
            ok = await self._preflight(stage_dir, timeout_s=max(3.0, timeout_s * 0.4))
            if not ok:
                log_warn("selfupdate.preflight_failed", dir=stage_dir)
                _jr("selfupdate.preflight_failed", stage="preflight", ok=False, dir=stage_dir)
                try:
                    _pub_evt("selfupdate.preflight_failed", {"dir": stage_dir})
                except Exception:
                    pass
                # Auto-rollback to LKG if configured
                if await self._try_rollback_to_lkg(timeout_s=timeout_s):
                    return
                return
            _jr("selfupdate.preflight_ok", stage="preflight", ok=True, dir=stage_dir)
            _set_det({"update": {"phase": "green_spawn"}})
            ok = await self._spawn_green_and_switch(stage_dir, timeout_s=timeout_s)
            if not ok:
                log_warn("selfupdate.green_failed", dir=stage_dir)
                _jr("selfupdate.green_failed", stage="green", ok=False, dir=stage_dir)
                try:
                    _pub_evt("selfupdate.green_failed", {"dir": stage_dir})
                except Exception:
                    pass
                if await self._try_rollback_to_lkg(timeout_s=timeout_s):
                    return
                return
            log_info("selfupdate.success", dir=stage_dir)
            _jr("selfupdate.success", stage="switch", ok=True, dir=stage_dir)
            try:
                _pub_evt("selfupdate.success", {"dir": stage_dir})
            except Exception:
                pass
        finally:
            try:
                _clear_det(); _clear_act()
            except Exception:
                pass

    async def _stage_code(self, source_dir: Optional[str]) -> Optional[str]:
        root = os.getcwd()
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(root, ".jinx", "stage", ts)
        pkg_src = source_dir or os.path.join(root, "jinx")
        try:
            os.makedirs(base, exist_ok=True)
            dst_pkg = os.path.join(base, "jinx")
            # Copytree with dirs_exist_ok for speed/idempotency
            await asyncio.to_thread(shutil.copytree, pkg_src, dst_pkg, dirs_exist_ok=True)
            # Copy entrypoint jinx.py for interpreter launch context
            src_entry = os.path.join(root, "jinx.py")
            if os.path.isfile(src_entry):
                await asyncio.to_thread(shutil.copy2, src_entry, os.path.join(base, "jinx.py"))
            # Hash staged content for traceability
            sha = await asyncio.to_thread(self._hash_dir, dst_pkg)
            _jr("selfupdate.staged_hash", stage="stage", ok=True, sha=sha)
            # Optional signature verification
            if not verify_stage_signature(base, sha):
                return None
            return base
        except Exception:
            return None

    def _hash_dir(self, dir_path: str) -> str:
        import hashlib
        h = hashlib.sha256()
        try:
            for root, _, files in os.walk(dir_path):
                for fn in sorted(files):
                    if fn.endswith((".pyc", ".pyo")):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, "rb") as f:
                            while True:
                                b = f.read(65536)
                                if not b:
                                    break
                                h.update(b)
                    except Exception:
                        continue
            return h.hexdigest()
        except Exception:
            return ""

    async def _preflight(self, stage_dir: str, *, timeout_s: float) -> bool:
        """Run isolated ensure_runtime to trigger auto-repairs, then smoke-imports."""
        py = sys.executable or "python"
        env = os.environ.copy()
        env["JINX_PACKAGE_DIR"] = os.path.join(stage_dir, "jinx")
        env["JINX_SELFUPDATE_MODE"] = "preflight"
        # Keep side effects minimal: disable heavy plugins and background tasks
        env["JINX_PLUGINS"] = "telemetry:off"
        env["JINX_MISSION_PLANNER_ENABLE"] = "0"
        env["JINX_API_ARCHITECT_ENABLE"] = "0"
        # Optional ephemeral venv
        try:
            if _SELFUPDATE_USE_VENV:
                _jr("selfupdate.venv_create", stage="preflight", ok=None)
                vpy = await create_venv(stage_dir)
                if vpy:
                    ok_install = await install_requirements(vpy, stage_dir, timeout_s=min(12.0, max(4.0, timeout_s * 0.5)))
                    if ok_install:
                        py = vpy
                        _jr("selfupdate.venv_ready", stage="preflight", ok=True)
                    else:
                        _jr("selfupdate.venv_req_failed", stage="preflight", ok=False)
                else:
                    _jr("selfupdate.venv_failed", stage="preflight", ok=False)
        except Exception:
            _jr("selfupdate.venv_error", stage="preflight", ok=False)
        # Phase 1: run ensure_runtime to perform repairs inside staged tree, and give tasks a short window
        code1 = (
            "import os, asyncio; "
            "from jinx.micro.runtime.resilience import install_resilience, schedule_repairs; install_resilience(); "
            "from jinx.micro.runtime.api import ensure_runtime; "
            "async def _main():\n"
            "    await ensure_runtime();\n"
            "    try:\n"
            "        await schedule_repairs()\n"
            "    except Exception:\n"
            "        pass\n"
            "    await asyncio.sleep(0.8)\n"
            "asyncio.run(_main()); "
            "print('ENSURE_OK')"
        )
        # Phase 2: smoke import core modules
        code2 = (
            "import importlib, os as _os, compileall; "
            "# Compile staged package to catch syntax errors early\n"
            "_pkg = _os.environ.get('JINX_PACKAGE_DIR','') or ''\n"
            "compileall.compile_dir(_pkg, force=False, quiet=1)\n"
            "import jinx; import jinx.orchestrator; import jinx.runtime_service; "
            "import jinx.micro.runtime.api; import jinx.micro.runtime.plugins; "
            "print('SMOKE_OK')"
        )
        try:
            # Run phase 1
            proc = await asyncio.create_subprocess_exec(
                py, "-c", code1,
                cwd=stage_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out1, err1 = await asyncio.wait_for(proc.communicate(), timeout=max(2.0, timeout_s * 0.6))
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                _jr("selfupdate.ensure_timeout", stage="preflight", ok=False)
                return False
            if not (proc.returncode == 0 and b"ENSURE_OK" in (out1 or b"")):
                _jr("selfupdate.ensure_failed", stage="preflight", ok=False)
                return False
            # Run phase 2
            proc2 = await asyncio.create_subprocess_exec(
                py, "-c", code2,
                cwd=stage_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out2, err2 = await asyncio.wait_for(proc2.communicate(), timeout=max(2.0, timeout_s * 0.4))
            except asyncio.TimeoutError:
                try:
                    proc2.kill()
                except Exception:
                    pass
                _jr("selfupdate.smoke_timeout", stage="preflight", ok=False)
                return False
            ok = bool(proc2.returncode == 0 and b"SMOKE_OK" in (out2 or b""))
            _jr("selfupdate.smoke_ok" if ok else "selfupdate.smoke_failed", stage="preflight", ok=ok)
            if not ok:
                return False
            # Phase 3: prompt render smoke to detect placeholder/ASCII issues
            code3 = (
                "import json;"
                "from jinx.prompts import render_prompt;"
                "shape='{\\n  \"name\": \"string\",\\n  \"resources\": [\\n    {\\n      \"name\": \"string\",\\n      \"fields\": {\"id\": \"int|str|float|bool\", ...},\\n      \"endpoints\": [\"list\", \"get\", \"create\", \"update\", \"delete\"]\\n    }\\n  ]\\n}';"
                "txt=render_prompt('architect_api', shape=shape, project_name='preflight', candidate_resources_json=json.dumps(['thing']), request='check');"
                "assert isinstance(txt,str) and txt.encode('ascii','ignore').decode('ascii')==txt;"
                "print('PROMPT_OK')"
            )
            proc3 = await asyncio.create_subprocess_exec(
                py, "-c", code3,
                cwd=stage_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out3, err3 = await asyncio.wait_for(proc3.communicate(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    proc3.kill()
                except Exception:
                    pass
                _jr("selfupdate.prompt_smoke_timeout", stage="preflight", ok=False)
                return False
            ok3 = bool(proc3.returncode == 0 and b"PROMPT_OK" in (out3 or b""))
            _jr("selfupdate.prompt_smoke_ok" if ok3 else "selfupdate.prompt_smoke_failed", stage="preflight", ok=ok3)
            return ok3
        except Exception:
            _jr("selfupdate.preflight_exception", stage="preflight", ok=False)
            return False

    async def _spawn_green_and_switch(self, stage_dir: str, *, timeout_s: float) -> bool:
        py = sys.executable or "python"
        env = os.environ.copy()
        env["JINX_PACKAGE_DIR"] = os.path.join(stage_dir, "jinx")
        # Handshake file
        hs_dir = os.path.join(os.getcwd(), ".jinx", "handshake")
        os.makedirs(hs_dir, exist_ok=True)
        hs_file = os.path.join(hs_dir, f"{int(time.time())}.json")
        env["JINX_SELFUPDATE_HANDSHAKE_FILE"] = hs_file
        env["JINX_SELFUPDATE_MODE"] = "green"
        # Prefer minimal side effects in initial boot window
        env["JINX_PLUGINS"] = env.get("JINX_PLUGINS", "")
        # Resource-constrained green for stabilization window
        env.setdefault("JINX_FRAME_MAX_CONC", "1")
        env.setdefault("JINX_GROUP_MAX_CONC", "1")
        env.setdefault("JINX_MAX_CONCURRENT", "2")
        # Shadow canary directory
        shadow_dir = os.path.join(stage_dir, ".jinx", "shadow")
        os.makedirs(shadow_dir, exist_ok=True)
        env["JINX_SELFUPDATE_SHADOW_DIR"] = shadow_dir
        proc = None
        try:
            import subprocess as _sp
            creationflags = 0
            try:
                if os.name == "nt":
                    # Lower green priority during stabilization
                    if _SELFUPDATE_GREEN_LOWPRIO:
                        for flag in (getattr(_sp, "BELOW_NORMAL_PRIORITY_CLASS", 0), getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0)):
                            creationflags |= int(flag or 0)
            except Exception:
                creationflags = 0
            proc = await asyncio.create_subprocess_exec(
                py, "jinx.py",
                cwd=stage_dir,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=creationflags if creationflags else 0,
            )
        except Exception:
            return False
        # Wait handshake: online then healthy
        deadline = time.time() + timeout_s
        online = False
        healthy = False
        while time.time() < deadline:
            await asyncio.sleep(0.3)
            try:
                with open(hs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                online = bool(data.get("online"))
                healthy = bool(data.get("healthy"))
            except Exception:
                continue
            if online and healthy:
                break
        if not (online and healthy):
            # Kill green and abort
            try:
                if proc and proc.returncode is None:
                    proc.kill()
            except Exception:
                pass
            _jr("selfupdate.handshake_failed", stage="green", ok=False)
            try:
                _pub_evt("selfupdate.handshake_failed", {"dir": stage_dir})
            except Exception:
                pass
            return False
        _jr("selfupdate.handshake_ok", stage="green", ok=True)
        # Optional shadow canary
        try:
            if _SELFUPDATE_CANARY:
                ok_can = await self._shadow_canary(shadow_dir, count=5, timeout_s=min(4.0, max(2.0, timeout_s * 0.25)))
                _jr("selfupdate.canary_ok" if ok_can else "selfupdate.canary_failed", stage="green", ok=ok_can)
                if not ok_can:
                    try:
                        if proc and proc.returncode is None:
                            proc.kill()
                    except Exception:
                        pass
                    return False
        except Exception:
            _jr("selfupdate.canary_error", stage="green", ok=False)
            return False
        # Green is healthy: stabilization window before shutting down blue
        stab_sec = _SELFUPDATE_STABILIZE_SEC
        t_end = time.time() + max(1.0, stab_sec)
        while time.time() < t_end:
            await asyncio.sleep(0.3)
            try:
                # If green died during stabilization, abort and keep blue
                if proc and proc.returncode is not None and proc.returncode != 0:
                    _jr("selfupdate.stabilize_green_died", stage="green", ok=False)
                    return False
            except Exception:
                pass
        _jr("selfupdate.stabilize_ok", stage="green", ok=True)
        # Stable: request graceful shutdown of blue with quiesce/drain
        await self._quiesce_and_drain()
        try:
            import jinx.state as jx_state
            jx_state.shutdown_event.set()
        except Exception:
            pass
        # Record LKG pointer
        try:
            self._write_lkg(stage_dir)
            _jr("selfupdate.lkg_set", stage="switch", ok=True)
        except Exception:
            _jr("selfupdate.lkg_failed", stage="switch", ok=False)
        return True

    async def _shadow_canary(self, shadow_dir: str, *, count: int, timeout_s: float) -> bool:
        # Write N request files and wait for corresponding .ok
        try:
            import uuid, time as _t
            deadline = _t.time() + timeout_s
            for i in range(max(1, count)):
                tid = uuid.uuid4().hex[:8]
                req = os.path.join(shadow_dir, f"{tid}.in")
                ack = os.path.join(shadow_dir, f"{tid}.ok")
                with open(req, "w", encoding="utf-8") as f:
                    f.write("ping")
                # wait for ack
                while _t.time() < deadline:
                    if os.path.isfile(ack):
                        break
                    await asyncio.sleep(0.1)
                else:
                    return False
            return True
        except Exception:
            return False

    async def _try_rollback_to_lkg(self, *, timeout_s: float) -> bool:
        try:
            if not _SELFUPDATE_AUTO_ROLLBACK:
                return False
            lkg = self._read_lkg()
            if not lkg:
                return False
            _jr("selfupdate.rollback_attempt", stage="rollback", ok=None)
            ok = await self._spawn_green_and_switch(lkg, timeout_s=timeout_s)
            _jr("selfupdate.rollback_ok" if ok else "selfupdate.rollback_failed", stage="rollback", ok=ok)
            return ok
        except Exception:
            _jr("selfupdate.rollback_error", stage="rollback", ok=False)
            return False

    def _read_lkg(self) -> Optional[str]:
        try:
            path = os.path.join(os.getcwd(), ".jinx", "selfupdate", "lkg.json")
            if not os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            d = str(data.get("dir") or "").strip()
            return d if d and os.path.isdir(d) else None
        except Exception:
            return None

    async def _canary(self, stage_dir: str, env: Dict[str, str], *, timeout_s: float) -> bool:
        py = (sys.executable or "python")
        code = (
            "import numpy as np; "
            "from jinx.micro.embeddings.vector_stage_semantic import _cosine_similarity as c; "
            "import numpy; "
            "a=np.array([1,0,0],dtype=np.float32); b=np.array([1,0,0],dtype=np.float32); "
            "print('OK' if abs(c(a,b)-1.0)<1e-6 else 'BAD')"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                py, "-c", code,
                cwd=stage_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            return proc.returncode == 0 and b"OK" in (out or b"")
        except Exception:
            return False

    async def _quiesce_and_drain(self) -> None:
        try:
            import jinx.state as jx_state
        except Exception:
            return
        # Quiesce: stop scheduling new turns
        try:
            jx_state.throttle_event.set()
        except Exception:
            pass
        # Drain: wait for active_turns to reach 0 or timeout
        drain_sec = _SELFUPDATE_DRAIN_SEC
        end_t = time.time() + max(1.0, drain_sec)
        while time.time() < end_t:
            try:
                cur = int(getattr(jx_state, "active_turns", 0) or 0)
                if cur <= 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)

    def _write_lkg(self, stage_dir: str) -> None:
        root = os.getcwd()
        d = os.path.join(root, ".jinx", "selfupdate")
        os.makedirs(d, exist_ok=True)
        rec = {"dir": stage_dir, "ts": time.time()}
        with open(os.path.join(d, "lkg.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)


async def spawn_self_update_manager() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(SelfUpdateManager())
