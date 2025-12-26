from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST

_REPROGRAM_BUDGET_MS = 2400
_SELFUPDATE_TIMEOUT_S = 18.0
_PLAN_TOPK = 8
_PLAN_EMBED_MS = 600
_PLAN_CG_WINDOWS = True
_PLAN_REFINE_MS = 500
_VERIFY_SANDBOX = True
_REPROGRAM_TESTS = True

_JSON_PLAN_EXAMPLE = r"""
{
  "patches": [
    {
      "path": "jinx/micro/llm/service.py",
      "strategy": "context",  
      "context_before": "def code_primer",
      "code": "# ... new implementation ..."
    }
  ]
}
"""


@dataclass
class PatchPlan:
    path: str
    strategy: str
    code: Optional[str] = None
    symbol: Optional[str] = None
    anchor: Optional[str] = None
    query: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    context_before: Optional[str] = None
    context_tolerance: Optional[float] = None


class SelfReprogrammer(MicroProgram):
    """Autonomous self-reprogramming orchestrator.

    Flow:
      - Accept `reprogram.request` with a natural language goal.
      - Ask LLM to synthesize a JSON patch plan (constrained schema).
      - Execute patches via autopatch with strict RT budgets and previews.
      - If the plan applies successfully (>=1 patch), trigger blue-green self-update.
    """

    def __init__(self) -> None:
        super().__init__(name="SelfReprogrammer")
        self._budget_ms = _REPROGRAM_BUDGET_MS
        self._apply_timeout_s = _SELFUPDATE_TIMEOUT_S
        self._sem = asyncio.Semaphore(1)
        self._embed_topk = _PLAN_TOPK
        self._embed_ms = _PLAN_EMBED_MS

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("self-reprogrammer online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        name = str(payload.get("name") or "")
        if name != "reprogram.request":
            return
        kwargs = payload.get("kwargs") or {}
        goal = str(kwargs.get("goal") or "").strip()
        if not goal:
            return
        try:
            timeout_s = float(kwargs.get("timeout_s") or self._apply_timeout_s)
        except Exception:
            timeout_s = self._apply_timeout_s
        await self._execute(goal=goal, timeout_s=timeout_s)

    async def _execute(self, *, goal: str, timeout_s: float) -> None:
        # Single-flight to avoid concurrent self-edits
        async with self._sem:
            # Begin transaction: snapshot potential targets up-front (best-effort)
            backup_map: dict[str, str] = {}
            try:
                plan = await asyncio.wait_for(self._synthesize_plan(goal), timeout=max(0.6, self._budget_ms / 1000.0))
            except Exception:
                plan = []
            if not plan:
                await self.log("no patch plan produced", level="warn")
                return
            # Pre-snapshot original files (ignore missing)
            try:
                root = os.getcwd()
                for p in plan:
                    if not p.path:
                        continue
                    ap = os.path.join(root, p.path)
                    if ap in backup_map:
                        continue
                    try:
                        with open(ap, "r", encoding="utf-8") as f:
                            backup_map[ap] = f.read()
                    except Exception:
                        backup_map[ap] = None  # type: ignore
            except Exception:
                pass
            ok_count = 0
            for p in plan:
                try:
                    applied = await asyncio.wait_for(self._apply_patch(p), timeout=max(0.6, self._budget_ms / 1000.0))
                except Exception:
                    applied = False
                if applied:
                    ok_count += 1
            if ok_count <= 0:
                await self.log("no patches applied", level="warn")
                return
            # Verify all modified targets before switch (best-effort aggregation)
            try:
                verified = await self._verify_modified_files()
            except Exception:
                verified = False
            if not verified:
                # Rollback all changed files
                await self._rollback_all(backup_map)
                await self.log("verification failed; skipping selfupdate", level="warn")
                return
            # Adversarial LLM tests (optional)
            tests_ok = await self._run_adversarial_tests(goal, budget_ms=800)
            if not tests_ok:
                await self._rollback_all(backup_map)
                await self.log("adversarial tests failed; rollback", level="warn")
                return
            # Trigger blue-green switch via existing self-update manager
            try:
                from jinx.micro.runtime.api import submit_task as _submit
                await _submit("selfupdate.apply", source_dir=None, timeout_s=timeout_s)
                await self.log("selfupdate.apply submitted")
            except Exception:
                await self.log("failed to submit selfupdate.apply", level="warn")

    async def _synthesize_plan(self, goal: str) -> List[PatchPlan]:
        """Ask OpenAI to produce a compact JSON patch plan under strict schema."""
        try:
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return []
        # Build embedding-driven context (files/snippets relevant to the goal)
        emb_ctx, emb_hits = await self._build_embedding_context(goal, topk=self._embed_topk, budget_ms=self._embed_ms)
        # Compose planner prompt with embedding context guidance (template)
        try:
            from jinx.prompts import render_prompt as _render_prompt
            # Risk policy text for planner guidance (deny globs)
            try:
                from jinx.micro.runtime.risk_policies import deny_patterns as _deny
                _den = _deny() or []
                risk_text = "DENY: " + (", ".join(_den) if _den else "(none)")
            except Exception:
                risk_text = "(no explicit policy)"
            prompt = _render_prompt(
                "planner_synthesize",
                goal=goal,
                embed_context=emb_ctx,
                topk=self._embed_topk,
                example_json=_JSON_PLAN_EXAMPLE,
                risk_text=risk_text,
            )
        except Exception:
            prompt = f"Goal: {goal}\n\n{emb_ctx}\n\n{_JSON_PLAN_EXAMPLE}"
        # Admission guard for LLM call
        try:
            from jinx.rt.admission import guard as _guard
        except Exception:
            _guard = None  # type: ignore
        try:
            if _guard is not None:
                async with _guard("llm", timeout_ms=200) as admitted:
                    if not admitted:
                        return []
                    out, _ = await _spark(prompt)
            else:
                out, _ = await _spark(prompt)
        except Exception:
            return []
        if not out:
            return []
        # Extract JSON
        data: Dict[str, Any] = {}
        try:
            import re
            m = re.search(r"\{[\s\S]*\}", out)
            s = m.group(0) if m else out.strip()
            data = json.loads(s)
        except Exception:
            return []
        patches_raw = list((data.get("patches") or []) if isinstance(data, dict) else [])
        plan: List[PatchPlan] = []
        for it in patches_raw:
            if not isinstance(it, dict):
                continue
            path = str(it.get("path") or "").strip()
            strategy = str(it.get("strategy") or "").strip().lower()
            if not path or not strategy:
                continue
            plan.append(PatchPlan(
                path=path,
                strategy=strategy,
                code=it.get("code"),
                symbol=it.get("symbol"),
                anchor=it.get("anchor"),
                query=it.get("query"),
                line_start=it.get("line_start"),
                line_end=it.get("line_end"),
                context_before=it.get("context_before"),
                context_tolerance=it.get("context_tolerance"),
            ))
        # Rerank plan by embedding relevance + strategy safety
        try:
            plan = await self._rerank_plan(plan, emb_hits)
        except Exception:
            pass
        # If plan does not cover top embedding files, ask for a quick refinement
        try:
            top_files = [f for (f, _sc) in emb_hits[: max(1, min(4, len(emb_hits)))]]
            covers = any(p.path in top_files for p in plan)
            if not covers:
                plan = await self._refine_plan_with_embeddings(goal, plan, top_files)
        except Exception:
            pass
        return plan

    async def _build_embedding_context(self, goal: str, *, topk: int, budget_ms: int) -> tuple[str, List[tuple[str, float]]]:
        """Return (context_text, hits) where hits are [(file_rel, score)]."""
        from jinx.rt.admission import guard as _guard
        hits: List[Dict[str, Any]] = []
        # Expand goal terms using brain concepts
        q = goal
        try:
            from jinx.micro.brain.concepts import activate_concepts as _brain_activate
            pairs = await _brain_activate(goal, top_k=max(4, topk))
            toks: List[str] = []
            seen: set[str] = set()
            for key, sc in pairs:
                low = (key or "").lower()
                if low.startswith("term: "):
                    t = low.split(": ", 1)[1]
                    if t and t not in seen:
                        toks.append(t)
                        seen.add(t)
                if len(toks) >= 6:
                    break
            if toks:
                q = (q + " " + " ".join(toks)).strip()
        except Exception:
            pass
        # Retrieve semantically relevant files
        from jinx.micro.embeddings.search_cache import search_project_cached as _search
        try:
            async with _guard("graph", timeout_ms=200):
                hits = await _search(q, k=max(1, topk), max_time_ms=budget_ms)
        except Exception:
            hits = []
        # Build context text
        ctx_lines: List[str] = []
        scores: List[tuple[str, float]] = []
        if hits:
            # Deduplicate by file, take the strongest hit per file
            best_by_file: Dict[str, Dict[str, Any]] = {}
            for h in hits:
                f = str(h.get("file") or "").strip()
                if not f:
                    continue
                s = float(h.get("score", 0.0) or 0.0)
                if f not in best_by_file or s > float(best_by_file[f].get("score", 0.0) or 0.0):
                    best_by_file[f] = h
            # Build snippets
            try:
                from jinx.micro.embeddings.project_lang import lang_for_file as _lang
            except Exception:
                _lang = None  # type: ignore
            for f, h in list(best_by_file.items())[:topk]:
                rel = f
                ls = int(h.get("line_start") or 1)
                le = int(h.get("line_end") or ls)
                abs_p = os.path.join(os.getcwd(), rel)
                snippet = ""
                file_text = ""
                try:
                    with open(abs_p, "r", encoding="utf-8", errors="ignore") as fh:
                        file_text = fh.read()
                        lines = file_text.splitlines()
                    a = max(1, ls - 6)
                    b = min(len(lines), le + 6)
                    snippet = "\n".join(lines[a - 1 : b])
                except Exception:
                    snippet = ""
                lang = ""
                try:
                    lang = _lang(rel) if _lang else ""
                except Exception:
                    lang = ""
                hdr = f"[CTX] [{rel}:{ls}-{le}]\n"
                block = f"```{lang}\n{snippet}\n```" if snippet else "(unreadable)"
                ctx_lines.append(hdr + block)
                scores.append((rel, float(h.get("score", 0.0) or 0.0)))
                # Optional: add small callgraph windows for Python symbol at hit line
                try:
                    if _PLAN_CG_WINDOWS and rel.endswith(".py") and file_text:
                        from jinx.micro.embeddings.project_py_scope import get_python_symbol_at_line as _sym_at  # type: ignore
                        from jinx.micro.embeddings.project_callgraph import windows_for_symbol as _cg_windows  # type: ignore
                        mid = int((ls + le) // 2)
                        sym, kind = _sym_at(file_text, mid)
                        if sym:
                            # Acquire RT guard and offload if available
                            from jinx.rt.admission import guard as _guard
                            try:
                                from jinx.rt.threadpool import run_cpu as _run_cpu  # type: ignore
                            except Exception:
                                _run_cpu = None  # type: ignore
                            wins: list[tuple[str, int, int, str]] = []
                            async with _guard("graph", timeout_ms=150):
                                try:
                                    if _run_cpu is not None:
                                        wins = await _run_cpu(_cg_windows, sym, prefer_rel=rel, callers_limit=1, callees_limit=1, around=8, scan_cap_files=80, time_budget_ms=300)
                                    else:
                                        import asyncio as _aio
                                        wins = await _aio.to_thread(_cg_windows, sym, None, rel, 1, 1, 8, 80, 300)  # type: ignore[arg-type]
                                except Exception:
                                    wins = []
                            for (rel2, a2, b2, kind2) in wins[:2]:
                                abs2 = os.path.join(os.getcwd(), rel2)
                                try:
                                    with open(abs2, "r", encoding="utf-8", errors="ignore") as fh2:
                                        lines2 = fh2.read().splitlines()
                                    a = max(1, int(a2) - 6)
                                    b = min(len(lines2), int(b2) + 6)
                                    sn2 = "\n".join(lines2[a - 1 : b])
                                except Exception:
                                    sn2 = ""
                                hdr2 = f"[CG {kind2}] [{rel2}:{a2}-{b2}]\n"
                                block2 = f"```py\n{sn2}\n```" if sn2 else "(unreadable)"
                                ctx_lines.append(hdr2 + block2)
                except Exception:
                    pass
        return ("\n\n".join(ctx_lines), scores)

    async def _rerank_plan(self, plan: List[PatchPlan], emb_hits: List[tuple[str, float]]) -> List[PatchPlan]:
        # Enforce risk policy: drop denied paths before scoring
        try:
            from jinx.micro.runtime.risk_policies import is_allowed_path as _risk_allow
        except Exception:
            def _risk_allow(_p: str) -> bool:  # type: ignore
                return True
        plan = [p for p in (plan or []) if (p and getattr(p, 'path', '') and _risk_allow(p.path))]
        if not plan or not emb_hits:
            return plan
        score_by_file = {f: sc for f, sc in emb_hits}
        # Strategy safety priors
        pref = {
            "symbol": 0.10,
            "symbol_body": 0.08,
            "context": 0.06,
            "semantic": 0.05,
            "anchor": 0.03,
            "line": 0.01,
            "write": -0.02,
            "codemod_rename": 0.07,
            "codemod_add_import": 0.06,
            "codemod_replace_import": 0.06,
        }
        def _score(p: PatchPlan) -> float:
            s_file = float(score_by_file.get(p.path, 0.0))
            s_strat = float(pref.get((p.strategy or "").lower(), 0.0))
            # tiny penalty for missing minimal fields
            miss_pen = 0.0
            if (p.strategy or "").lower() in ("symbol", "symbol_body") and not p.symbol:
                miss_pen += 0.02
            if (p.strategy or "").lower() == "context" and not (p.context_before and p.code):
                miss_pen += 0.02
            return 1.0 * s_file + s_strat - miss_pen
        plan.sort(key=_score, reverse=True)
        return plan

    async def _refine_plan_with_embeddings(self, goal: str, plan: List[PatchPlan], top_files: List[str]) -> List[PatchPlan]:
        """Ask LLM to rewrite the plan to target the most relevant files (embedding top list).
        Keeps same schema. Best-effort within a small time budget.
        """
        try:
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return plan
        budget_ms = _PLAN_REFINE_MS
        try:
            js = json.dumps({"patches": [p.__dict__ for p in plan]}, ensure_ascii=False)
        except Exception:
            js = "{}"
        try:
            from jinx.prompts import render_prompt as _render_prompt
            prompt = _render_prompt("planner_refine_embed", goal=goal, top_files_csv=", ".join(top_files), current_plan_json=js)
        except Exception:
            prompt = f"Goal: {goal}\nFiles: {', '.join(top_files)}\n{js}"
        # Admission guard for LLM call
        try:
            from jinx.rt.admission import guard as _guard
        except Exception:
            _guard = None  # type: ignore
        try:
            if _guard is not None:
                async with _guard("llm", timeout_ms=150) as admitted:
                    if not admitted:
                        return plan
                    out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.1, budget_ms / 1000.0))
            else:
                out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.1, budget_ms / 1000.0))
        except Exception:
            return plan
        if not out:
            return plan
        try:
            import re
            m = re.search(r"\{[\s\S]*\}", out)
            s = m.group(0) if m else out.strip()
            data = json.loads(s)
        except Exception:
            return plan
        patches_raw = list((data.get("patches") or []) if isinstance(data, dict) else [])
        new_plan: List[PatchPlan] = []
        for it in patches_raw:
            if not isinstance(it, dict):
                continue
            path = str(it.get("path") or "").strip()
            strategy = str(it.get("strategy") or "").strip().lower()
            if not path or not strategy:
                continue
            new_plan.append(PatchPlan(
                path=path,
                strategy=strategy,
                code=it.get("code"),
                symbol=it.get("symbol"),
                anchor=it.get("anchor"),
                query=it.get("query"),
                line_start=it.get("line_start"),
                line_end=it.get("line_end"),
                context_before=it.get("context_before"),
                context_tolerance=it.get("context_tolerance"),
            ))
        return new_plan or plan

    async def _apply_patch(self, p: PatchPlan) -> bool:
        """Apply a single patch through autopatch with preview+commit.
        Supports strategy 'codemod_rename' via libcst.
        Safely rollbacks if verification fails.
        """
        root = os.getcwd()
        abs_path = os.path.join(root, p.path) if p.path else None
        # Risk policy: skip denied paths early
        try:
            from jinx.micro.runtime.risk_policies import is_allowed_path as _risk_allow
        except Exception:
            def _risk_allow(_p: str) -> bool:  # type: ignore
                return True
        if p.path and not _risk_allow(p.path):
            return False
        # Handle codemod rename separately
        if (p.strategy or "").lower() == "codemod_rename" and abs_path and p.symbol and p.code:
            try:
                from jinx.codemods.rename_symbol import rename_symbol_file as _rename
            except Exception:
                return False
            try:
                # Backup
                original = ""
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        original = f.read()
                except Exception:
                    original = ""
                ok = await _rename(abs_path, old_name=p.symbol, new_name=p.code)
                if not ok:
                    return False
                # Verify single file
                v_ok = await self._verify_code_text(original=None, new_text_path=abs_path)
                if not v_ok:
                    # rollback
                    try:
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(original)
                    except Exception:
                        pass
                    return False
                self._track_modified(abs_path)
                return True
            except Exception:
                return False
        # Codemod: add import (symbol=module, code=name, anchor=alias)
        if (p.strategy or "").lower() == "codemod_add_import" and abs_path and (p.symbol or p.code):
            try:
                from jinx.codemods.imports import add_import_to_file as _add_import
            except Exception:
                return False
            ok = await _add_import(abs_path, p.symbol or "", name=(p.code or None), alias=(p.anchor or None))
            if not ok:
                return False
            v_ok = await self._verify_code_text(original=None, new_text_path=abs_path)
            if not v_ok:
                return False
            self._track_modified(abs_path)
            return True

        # Codemod: replace import module (symbol=old_module, code=new_module)
        if (p.strategy or "").lower() == "codemod_replace_import" and abs_path and (p.symbol and p.code):
            try:
                from jinx.codemods.imports import replace_import_in_file as _rep_import
            except Exception:
                return False
            ok = await _rep_import(abs_path, p.symbol, p.code)
            if not ok:
                return False
            v_ok = await self._verify_code_text(original=None, new_text_path=abs_path)
            if not v_ok:
                return False
            self._track_modified(abs_path)
            return True

        # Default path: autopatch
        from jinx.micro.runtime.patch.autopatch import AutoPatchArgs, autopatch
        args = AutoPatchArgs(
            path=abs_path,
            code=p.code,
            line_start=int(p.line_start) if p.line_start else None,
            line_end=int(p.line_end) if p.line_end else None,
            symbol=p.symbol,
            anchor=p.anchor,
            query=p.query,
            preview=False,
            context_before=p.context_before,
            context_tolerance=float(p.context_tolerance) if p.context_tolerance else None,
        )
        # Read original before commit for rollback
        original = ""
        if abs_path:
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    original = f.read()
            except Exception:
                original = ""
        # Preview
        ok_prev, strat, diff = await autopatch(AutoPatchArgs(**{**args.__dict__, "preview": True}))  # type: ignore[arg-type]
        if not ok_prev:
            return False
        # Commit
        ok_commit, strat2, detail = await autopatch(args)
        if not ok_commit:
            return False
        # Verify affected files parsed from diff
        files = self._files_from_diff(detail or "", default_path=abs_path)
        v_all = True
        for fp in files:
            ok_one = await self._verify_code_text(original=None, new_text_path=fp)
            if not ok_one:
                v_all = False
                break
        if not v_all and abs_path and original:
            try:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(original)
            except Exception:
                pass
            return False
        for fp in files:
            self._track_modified(fp)
        return True

    # --- Verification helpers ---
    def _files_from_diff(self, diff: str, default_path: str | None) -> list[str]:
        out: list[str] = []
        if not diff and default_path:
            return [default_path]
        for ln in (diff or "").splitlines():
            if ln.startswith("+++ "):
                # +++ b/<file>
                parts = ln.split()
                if len(parts) >= 2:
                    p = parts[1]
                    if p.startswith("b/"):
                        p = p[2:]
                    out.append(os.path.join(os.getcwd(), p))
        if not out and default_path:
            out = [default_path]
        return out

    async def _verify_code_text(self, original: Optional[str], new_text_path: str) -> bool:
        try:
            with open(new_text_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception:
            return False
        # AST & libcst checks
        try:
            from jinx.verify.ast_checks import syntax_ok, libcst_ok
            ok1, _ = syntax_ok(code)
            ok2, _ = libcst_ok(code)
            if not (ok1 and ok2):
                return False
        except Exception:
            return False
        # py_compile smoke
        try:
            import py_compile
            py_compile.compile(new_text_path, doraise=True)
        except Exception:
            return False
        # Hypothesis properties (optional)
        try:
            from jinx.verify.properties import run_basic_properties
            okp, _ = run_basic_properties(code)
            if not okp:
                return False
        except Exception:
            # If hypothesis not present, run_basic_properties returns soft pass; errors here are hard fail
            return False
        # Z3 invariants (optional)
        try:
            from jinx.verify.z3_invariants import check_invariants
            okz, _ = check_invariants(code, file_path=new_text_path)
            if not okz:
                return False
        except Exception:
            # If z3 not present, check_invariants soft-passes; errors here are hard fail
            return False
        # Optional sandbox smoke
        try:
            if _VERIFY_SANDBOX:
                from jinx.sandbox.async_runner import run_sandbox
                await run_sandbox("print('verify-ok')")
        except Exception:
            # Non-fatal; sandbox smoke optional
            pass
        return True

    def _track_modified(self, path: str) -> None:
        try:
            import jinx.state as jx_state
            mods = list(getattr(jx_state, "reprogram_modified", []) or [])
            if path not in mods:
                mods.append(path)
            setattr(jx_state, "reprogram_modified", mods)
        except Exception:
            pass

    async def _verify_modified_files(self) -> bool:
        try:
            import jinx.state as jx_state
            files = list(getattr(jx_state, "reprogram_modified", []) or [])
        except Exception:
            files = []
        if not files:
            return True
        for fp in files:
            ok = await self._verify_code_text(original=None, new_text_path=fp)
            if not ok:
                return False
        return True

    async def _rollback_all(self, backups: dict[str, str]) -> None:
        try:
            for ap, src in backups.items():
                if src is None:
                    # File wasn't readable before; skip destructive writes
                    continue
                try:
                    with open(ap, "w", encoding="utf-8") as f:
                        f.write(src)
                except Exception:
                    pass
        except Exception:
            pass

    async def _run_adversarial_tests(self, goal: str, *, budget_ms: int = 800) -> bool:
        """Ask LLM to synthesize a tiny adversarial test for recent changes and run it in sandbox.
        If disabled or on error, returns True. Best-effort within time budget.
        """
        try:
            if not _REPROGRAM_TESTS:
                return True
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return True
        try:
            from jinx.prompts import render_prompt as _render_prompt
            prompt = _render_prompt("reprogram_adversarial", goal=goal)
        except Exception:
            prompt = f"Goal: {goal}"
        # Admission guard for LLM call
        try:
            from jinx.rt.admission import guard as _guard
        except Exception:
            _guard = None  # type: ignore
        try:
            if _guard is not None:
                async with _guard("llm", timeout_ms=150) as admitted:
                    if not admitted:
                        return True
                    out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.1, budget_ms / 1000.0))
            else:
                out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.1, budget_ms / 1000.0))
        except Exception:
            return True
        if not out:
            return True
        # Extract code fence or fallback to text
        code = out
        try:
            import re as _re
            m = _re.search(r"```(?:python)?\n([\s\S]*?)\n```", out)
            if m:
                code = m.group(1)
        except Exception:
            pass
        # Run in sandbox
        try:
            from jinx.sandbox.async_runner import run_sandbox
            err_box = {"err": None}
            async def _cb(e):
                err_box["err"] = e
            await run_sandbox(code, callback=_cb)
            return (err_box["err"] is None)
        except Exception:
            return True


async def spawn_self_reprogrammer() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(SelfReprogrammer())
