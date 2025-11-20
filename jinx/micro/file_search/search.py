from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from jinx.micro.protocol.common import (
    FuzzyFileSearchResult,
)
from jinx.micro.text.fuzzy import fuzzy_match


_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__"}


@dataclass
class SearchConfig:
    limit_per_root: int = 50
    max_threads: int = 12
    compute_indices: bool = True
    follow_symlinks: bool = False


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".")


def _scan_root(
    root: str,
    query: str,
    config: SearchConfig,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[FuzzyFileSearchResult]:
    results: List[Tuple[int, FuzzyFileSearchResult]] = []  # (score, result)
    root_path = Path(root)
    if not root_path.exists():
        return []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=config.follow_symlinks):
        # prune dirs
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        if cancelled and cancelled():
            break

        for fn in filenames:
            if cancelled and cancelled():
                break
            p = Path(dirpath) / fn
            rel = str(p.as_posix())
            m = fuzzy_match(rel, query)
            if m is None:
                continue
            indices, score = m
            # keep lower score first (better)
            res = FuzzyFileSearchResult(
                root=str(root_path.as_posix()),
                path=rel,
                file_name=p.name,
                score=int(score),
                indices=indices if config.compute_indices else None,
            )
            results.append((score, res))

    # sort by score asc (better), then path asc
    results.sort(key=lambda x: (x[0], x[1].path))
    # trim per limit
    trimmed = [r for _, r in results[: config.limit_per_root]]
    return trimmed


def run_fuzzy_file_search(
    query: str,
    roots: Iterable[str],
    cancellation_flag: Optional[threading.Event] = None,
    *,
    limit_per_root: int = 50,
    max_threads: Optional[int] = None,
    compute_indices: bool = True,
) -> List[FuzzyFileSearchResult]:
    """Search files under roots, returning best fuzzy matches per root.

    - Uses simple pruning and a thread pool per root.
    - Cancellation supported via threading.Event.
    """
    roots_list = [str(r) for r in roots]
    if not roots_list:
        return []

    cfg = SearchConfig(
        limit_per_root=limit_per_root,
        max_threads=max_threads or os.cpu_count() or 4,
        compute_indices=compute_indices,
    )

    def is_cancelled() -> bool:
        return bool(cancellation_flag and cancellation_flag.is_set())

    out: List[FuzzyFileSearchResult] = []

    threads = max(1, min(cfg.max_threads, len(roots_list)))
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {
            ex.submit(_scan_root, root, query, cfg, is_cancelled): root for root in roots_list
        }
        for fut in as_completed(futures):
            if is_cancelled():
                break
            try:
                out.extend(fut.result())
            except Exception:
                # best-effort: ignore root failures
                pass

    # global ordering: score asc, then path asc
    out.sort(key=lambda r: (r.score, r.path))
    return out
