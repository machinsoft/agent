from __future__ import annotations

import re
import json as _json
from typing import Dict, Tuple
import hashlib as _hashlib

# Compact and budget-aware formatter for <embeddings_*> blocks.
# Goals:
# - Preserve tags and relative order but aggressively trim internals.
# - Enforce hard character budgets per block class with tunable weights.
# - Reduce noise: dedupe headers, collapse whitespace, trim code fences, shorten floats.

_TAGS = (
    ("embeddings_code", "code"),
    ("embeddings_refs", "refs"),
    ("embeddings_graph", "graph"),
    ("embeddings_memory", "memory"),
    ("embeddings_brain", "brain"),
    ("embeddings_meta", "meta"),  # appended by compactor when used
)

_BLOCK_RE: Dict[str, re.Pattern[str]] = {
    tag: re.compile(rf"(?s)<{tag}>(.*?)</{tag}>") for tag, _ in _TAGS
}


def _parse_blocks(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = text or ""
    for tag, _ in _TAGS:
        m = _BLOCK_RE[tag].search(s)
        if m:
            out[tag] = m.group(1) or ""
    return out


def _build_global_claims(comp_map: Dict[str, str]) -> list[str]:
    """Global top tokens across all blocks for a single-line bias claim.

    Produces one line like:
      Z= P3(7) T1(5) S2(3)
    Gate with JINX_CTX_META_GLOBAL_CLAIMS (default on).
    """
    top_n = 6
    toks_all: list[str] = []
    for tag, _label in _TAGS:
        s = (comp_map.get(tag) or "")
        if not s:
            continue
        toks_all.extend(re.findall(r"\b([PSTFIE]\d+)\b", s))
    if not toks_all:
        return []
    cnt: Dict[str, int] = {}
    for t in toks_all:
        cnt[t] = cnt.get(t, 0) + 1
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    if not items:
        return []
    body = " ".join([f"{t}({n})" for t, n in items])
    return [f"Z= {body}"]


def _build_weight_claims(comp_map: Dict[str, str]) -> list[str]:
    """Weighted token exposure across all compacted blocks (normalized).

    Emits a single line:
      W= P3:0.42 T1:0.31 S2:0.17
    Gate with JINX_CTX_META_WEIGHT_CLAIMS (default off to avoid redundancy).
    """
    return []
    cnt: Dict[str, int] = {}
    for tag, _label in _TAGS:
        s = (comp_map.get(tag) or "")
        if not s:
            continue
        for t in re.findall(r"\b([PSTFIE]\d+)\b", s):
            cnt[t] = cnt.get(t, 0) + 1
    if not cnt:
        return []
    total = sum(cnt.values()) or 1
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    body = " ".join([f"{t}:{cnt_t/total:.2f}" for t, cnt_t in items])
    return [f"W= {body}"]


def _clean_ws(body: str) -> str:
    # Collapse repeated blank lines and strip trailing spaces
    lines = [(ln.rstrip()) for ln in (body or "").splitlines()]
    out = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()


def _trim_floats(s: str) -> str:
    # Shorten floats like 0.123456 -> 0.12 within parentheses or weights
    return re.sub(r"(\d+\.\d{2})\d+", r"\1", s)


def _minify_code(body: str) -> str:
    """Heuristic code minifier: remove obvious comments/blank lines and long trailing whitespace.
    Conservative to avoid mangling meaning across languages.
    """
    lines = []
    skip_next_blank = False
    for raw in (body or "").splitlines():
        ln = raw.rstrip()
        lstr = ln.lstrip()
        # Drop common comment-only lines quickly
        if lstr.startswith("#") or lstr.startswith("//") or lstr.startswith("/*") or lstr == "*/":
            continue
        # Collapse docstrings/blocks only if line is triple-quote alone
        if lstr in ('"""', "'''"):
            # keep one boundary and drop subsequent empty
            if lines and lines[-1] == lstr:
                continue
        if lstr.strip() == "":
            if skip_next_blank:
                continue
            skip_next_blank = True
            lines.append("")
            continue
        skip_next_blank = False
        # Clamp very long lines to reduce token cost
        if len(ln) > 400:
            ln = ln[:400]
        lines.append(ln)
    # Drop leading/trailing blanks
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _extract_salient(present: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Extract salient tokens from <embeddings_brain> and count mentions per block.

    Returns (weight_boost_per_tag, mentions_per_tag) where weight_boost_per_tag are relative factors.
    """
    brain = (present.get("embeddings_brain") or "").lower()
    # Parse tokens like "term: foo", "path: file", "symbol: name" or generic lines before (score)
    toks = []
    for m in re.finditer(r"(?mi)\b(term|path|symbol|framework|import|error):\s*([^\n\r\(]+)", brain):
        toks.append((m.group(1) or "").strip().lower() + ":" + (m.group(2) or "").strip().lower())
    if not toks:
        # Fallback: capture tokens before a score in parentheses
        for m in re.finditer(r"(?m)^\s*\-\s*([^\(\n\r]+)\s*\((?:\d+|\d+\.\d+)\)\s*$", brain):
            t = (m.group(1) or "").strip().lower()
            if t:
                toks.append(t)
    if not toks:
        return {}, {}
    # Count mentions in each block
    tags = ["embeddings_code","embeddings_refs","embeddings_graph","embeddings_memory"]
    mentions: Dict[str, int] = {t: 0 for t in tags}
    for t in tags:
        body = (present.get(t) or "").lower()
        if not body:
            continue
        c = 0
        for tok in toks:
            try:
                c += len(list(re.finditer(re.escape(tok), body)))
            except Exception:
                continue
        mentions[t] = c
    total_m = sum(mentions.values()) or 0
    if total_m <= 0:
        return {}, mentions
    # Convert to boost factors
    gamma = 0.5
    boost = {t: 1.0 + gamma * (mentions[t] / float(total_m)) for t in mentions}
    return boost, mentions


def _graph_edge_density_boost(present: Dict[str, str]) -> float:
    body = (present.get("embeddings_graph") or "")
    if not body.strip():
        return 1.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    edges = sum(1 for ln in lines if "->" in ln)
    dens = min(1.0, max(0.0, edges / max(1.0, float(len(lines)))))
    gamma = 0.35
    return 1.0 + gamma * dens


def _budget_plan(present: Dict[str, str], total_budget: int) -> Dict[str, int]:
    # Default weights can be tuned via env; code highest priority
    w_code, w_refs, w_graph, w_mem, w_brain = 0.5, 0.25, 0.1, 0.12, 0.03
    weights: Dict[str, float] = {
        "embeddings_code": w_code,
        "embeddings_refs": w_refs,
        "embeddings_graph": w_graph,
        "embeddings_memory": w_mem,
        "embeddings_brain": w_brain,
        "embeddings_meta": 0.03,
    }
    # Presence-aware normalization with minimum floor
    floors = {
        "embeddings_code": 400,
        "embeddings_refs": 200,
        "embeddings_graph": 120,
        "embeddings_memory": 200,
        "embeddings_brain": 80,
        "embeddings_meta": 60,
    }
    # Optional dynamic reweighting based on salient tokens from brain
    dyn_on = True
    if dyn_on:
        boost, _mentions = _extract_salient(present)
        for t, f in boost.items():
            if t in weights:
                weights[t] *= max(0.1, float(f or 1.0))
    # Graph density boost (adaptive from KG edges)
    weights["embeddings_graph"] *= _graph_edge_density_boost(present)
    # Keep minimal budget for present blocks
    present_tags = [tag for tag in weights.keys() if tag in present and present[tag].strip()]
    if not present_tags:
        return {}
    w_sum = sum(weights[t] for t in present_tags)
    budgets: Dict[str, int] = {}
    for t in present_tags:
        frac = (weights[t] / w_sum) if w_sum > 0 else (1.0 / len(present_tags))
        raw = int(max(floors.get(t, 0), frac * total_budget))
        budgets[t] = raw
    return budgets


def _clamp(body: str, budget: int) -> str:
    s = (body or "").strip()
    if budget <= 0 or len(s) <= budget:
        return s
    return s[:budget]


def _compact_one(tag: str, body: str, budget: int) -> str:
    b = _clean_ws(body)
    if tag == "embeddings_code":
        b = _minify_code(b)
    elif tag == "embeddings_refs":
        # Drop code fences language to save space; keep markers compact
        b = re.sub(r"```[a-zA-Z0-9_\-]*\n", "```\n", b)
        b = _trim_floats(b)
    elif tag == "embeddings_graph":
        # Compress edges: collapse arrows/spaces
        b = re.sub(r"\s*\n\s*", "\n", b)
        b = re.sub(r"\s*->\s*", "->", b)
        b = re.sub(r"\s*:\s*", ":", b)
    elif tag == "embeddings_memory":
        # Lines already pre-trimmed; collapse whitespace further
        b = re.sub(r"\s+", " ", b)
        b = re.sub(r"\s*\n\s*", "\n", b)
    elif tag == "embeddings_brain":
        b = _trim_floats(b)
        # Convert to compact list, one per line without bullets
        b = re.sub(r"^\s*\-\s*", "", b, flags=re.MULTILINE)
    return _clamp(b, budget)


def _replace_once(text: str, tag: str, body: str) -> str:
    rx = _BLOCK_RE[tag]
    return rx.sub(lambda m: f"<{tag}>\n{body}\n</{tag}>", text, count=1)


def _brain_tokens(present: Dict[str, str]) -> list[str]:
    brain = (present.get("embeddings_brain") or "").lower()
    toks: list[str] = []
    for m in re.finditer(r"(?mi)\b(term|path|symbol|framework|import|error):\s*([^\n\r\(]+)", brain):
        t = (m.group(2) or "").strip().lower()
        if t:
            toks.append(t)
    return toks[:64]


def _code_ranges_only(present: Dict[str, str], comp_map: Dict[str, str]) -> Tuple[Dict[str, str], list[str]]:
    """Replace code fences with compact range references and produce claim headers.

    Returns (updated_comp_map, claim_lines_for_meta)
    """
    code_raw = (present.get("embeddings_code") or "")
    if not code_raw.strip():
        return comp_map, []
    # Parse headers like [rel:ls-le] followed by a code fence
    lines = code_raw.splitlines()
    i = 0
    ranges: list[Tuple[str, int, int, str]] = []  # (rel, ls, le, code_str)
    while i < len(lines):
        m = re.match(r"^\[(.+?):(\d+)-(\d+)\]\s*$", lines[i].strip())
        if not m:
            i += 1
            continue
        rel = (m.group(1) or "").strip()
        ls = int(m.group(2) or 0)
        le = int(m.group(3) or 0)
        # look for fence start
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("```"):
            j += 1
        if j >= len(lines):
            i += 1
            continue
        # fence end
        k = j + 1
        while k < len(lines) and not lines[k].strip().startswith("```"):
            k += 1
        code_str = "\n".join(lines[j+1:k]) if k < len(lines) else ""
        ranges.append((rel, ls, le, code_str))
        i = k + 1 if k < len(lines) else j + 1

    if not ranges:
        return comp_map, []
    # Build path tokens and claims
    claims: list[str] = []
    toks = _brain_tokens(present)
    hs_max = 2
    hs_chars = 80
    path_token: Dict[str, str] = {}
    pcount = 0
    # Attempt to reuse existing P tokens from meta
    meta = (present.get("embeddings_meta") or "")
    for mm in re.finditer(r"(?m)^P(\d+)\s*=\s*path:\s*(.+)$", meta):
        try:
            idx = int(mm.group(1) or 0)
            pcount = max(pcount, idx)
            path = (mm.group(2) or "").strip()
            path_token[path] = f"P{idx}"
        except Exception:
            continue
    lines_out: list[str] = []
    for rel, ls, le, body in ranges:
        if rel not in path_token:
            pcount += 1
            path_token[rel] = f"P{pcount}"
        px = path_token[rel]
        # Compute short sha of code body
        sha = _hashlib.sha1(body.encode("utf-8", errors="ignore")).hexdigest()[:8]
        # Hotspot lines: pick lines containing salient tokens
        hs_lines: list[str] = []
        if toks and hs_max > 0:
            for ln in body.splitlines():
                low = ln.lower()
                if any(t in low for t in toks):
                    hs_lines.append(ln.strip()[:hs_chars])
                    if len(hs_lines) >= hs_max:
                        break
        ref = f"{px}:{ls}-{le} [sha={sha}]"
        if hs_lines:
            ref = ref + " | " + " ; ".join(hs_lines)
        lines_out.append(ref)
        claims.append(f"C{len(claims)+1}={px}:{ls}-{le},hs={len(hs_lines)}")
    # Update comp_map and append mapping and claims into meta by caller
    comp_map["embeddings_code"] = "\n".join(lines_out)
    # Build mapping lines for any new Px
    mapping_lines = [f"{tok}=path: {pth}" for pth, tok in path_token.items() if (f"{tok}=path: {pth}") not in meta]
    claims = mapping_lines + claims
    return comp_map, claims


def _build_cross_claims(comp_map: Dict[str, str]) -> list[str]:
    """Summarize token usage across refs/graph/memory as compact meta claims.

    Produces lines like:
      R= P3(2) T1(4)
      G= P3(1)
      M= S2(3) E1(1)
    Gate with JINX_CTX_META_CLAIMS_EXT (default on). Limit items per block with JINX_CTX_META_CLAIMS_TOP.
    """
    top_n = 6
    out: list[str] = []
    for tag, alias in (("embeddings_refs", "R"), ("embeddings_graph", "G"), ("embeddings_memory", "M")):
        s = (comp_map.get(tag) or "")
        if not s.strip():
            continue
        toks = re.findall(r"\b([PSTFIE]\d+)\b", s)
        if not toks:
            continue
        cnt: Dict[str, int] = {}
        for t in toks:
            cnt[t] = cnt.get(t, 0) + 1
        items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        if not items:
            continue
        body = " ".join([f"{t}({n})" for t, n in items])
        out.append(f"{alias}= {body}")
    return out


def _token_map_compress(all_blocks: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    """Build a short token mapping for frequent long keys and replace them across blocks.

    Targets: path:, symbol:, term:, framework:, import:, error: lines.
    Returns (new_blocks, meta_str) where meta_str contains mapping lines for <embeddings_meta>.
    """
    text = "\n".join([all_blocks.get(t, "") for t, _ in _TAGS if t != "embeddings_meta"])  # original order concat
    pats = {
        "path": re.compile(r"(?mi)\bpath:\s*([^\n\r]+)"),
        "symbol": re.compile(r"(?mi)\bsymbol:\s*([^\n\r]+)"),
        "term": re.compile(r"(?mi)\bterm:\s*([^\n\r]+)"),
        "framework": re.compile(r"(?mi)\bframework:\s*([^\n\r]+)"),
        "import": re.compile(r"(?mi)\bimport:\s*([^\n\r]+)"),
        "error": re.compile(r"(?mi)\berror:\s*([^\n\r]+)"),
    }
    # Count frequencies
    freq: Dict[Tuple[str, str], int] = {}
    for kind, rx in pats.items():
        for m in rx.finditer(text):
            key = (kind, (m.group(1) or "").strip())
            if key[1]:
                freq[key] = freq.get(key, 0) + 1
    if not freq:
        return all_blocks, ""
    # Rank by potential savings: len(label) * (count-1)
    ranked = sorted(freq.items(), key=lambda kv: (len(kv[0][1]) * max(0, kv[1] - 1)), reverse=True)
    # Cap mapping size
    cap = 32
    mapping: Dict[Tuple[str, str], str] = {}
    counters = {"path": 0, "symbol": 0, "term": 0, "framework": 0, "import": 0, "error": 0}
    for (kind, label), _n in ranked:
        if len(mapping) >= cap:
            break
        counters[kind] += 1
        tok = {
            "path": f"P{counters['path']}",
            "symbol": f"S{counters['symbol']}",
            "term": f"T{counters['term']}",
            "framework": f"F{counters['framework']}",
            "import": f"I{counters['import']}",
            "error": f"E{counters['error']}",
        }[kind]
        mapping[(kind, label)] = tok
    if not mapping:
        return all_blocks, ""
    # Replace in each block
    new_blocks: Dict[str, str] = {}
    for tag, _label in _TAGS:
        if tag == "embeddings_meta":
            continue
        b = all_blocks.get(tag, "")
        if not b:
            continue
        for (kind, label), tok in mapping.items():
            b = re.sub(rf"(?mi)\b{kind}:\s*{re.escape(label)}\b", tok, b)
        new_blocks[tag] = b
    # Build meta body
    meta_lines = []
    for (kind, label), tok in mapping.items():
        meta_lines.append(f"{tok}={kind}:{label}")
    meta_str = "\n".join(meta_lines)
    return new_blocks, meta_str


def _parse_path_tokens(meta_str: str) -> Dict[str, str]:
    """Extract mapping of rel path -> P# token from <embeddings_meta>."""
    out: Dict[str, str] = {}
    s = meta_str or ""
    for m in re.finditer(r"(?m)^P(\d+)\s*=\s*path:\s*([^\n\r]+)$", s):
        try:
            pnum = int(m.group(1) or 0)
        except Exception:
            continue
        rel = (m.group(2) or "").strip()
        if not rel:
            continue
        tok = f"P{pnum}"
        out[rel] = tok
        # Also store normalized slashed version
        out[rel.replace("\\", "/")] = tok
    return out


def _load_symbol_index_sync() -> Dict[str, object]:
    """Best-effort load of symbol_index.json without async dependencies."""
    try:
        path = os.path.join("emb", "_state", "symbol_index.json")
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f) or {}
    except Exception:
        return {}


def _build_relation_claims(present: Dict[str, str], comp_map: Dict[str, str]) -> list[str]:
    """Build relation claims L= using symbol index.

    Format (one per symbol):
      L name=<sym> D= P3 P5 C= P2 P2
    Gate with JINX_CTX_META_REL_CLAIMS (default on).
    """
    on = True
    meta_prev = (present.get("embeddings_meta") or "")
    pmap = _parse_path_tokens(meta_prev)
    if not pmap:
        return []
    # symbols from brain
    brain = (present.get("embeddings_brain") or "")
    syms: list[str] = []
    for m in re.finditer(r"(?mi)\bsymbol:\s*([^\n\r\(]+)", brain):
        nm = (m.group(1) or "").strip()
        if nm and nm not in syms:
            syms.append(nm)
    if not syms:
        return []
    topn = 4
    capk = 4
    idx = _load_symbol_index_sync()
    defs_map: Dict[str, list] = (idx.get("defs") or {})  # type: ignore[assignment]
    calls_map: Dict[str, list] = (idx.get("calls") or {})  # type: ignore[assignment]
    claims: list[str] = []
    for name in syms[:topn]:
        defs = defs_map.get(name) or []
        calls = calls_map.get(name) or []
        # Map rel paths to P# tokens
        def _to_p(rel_line) -> str:
            try:
                rel = (rel_line[0] if isinstance(rel_line, (list, tuple)) else "") or ""
                tok = pmap.get(rel) or pmap.get(rel.replace("\\", "/"))
                return tok or ""
            except Exception:
                return ""
        d_p = [t for t in [
            _to_p(x) for x in defs[:capk]
        ] if t]
        c_p = [t for t in [
            _to_p(x) for x in calls[:capk]
        ] if t]
        if not d_p and not c_p:
            continue
        line = f"L name={name}"
        if d_p:
            line += " D= " + " ".join(d_p)
        if c_p:
            line += " C= " + " ".join(c_p)
        claims.append(line)
    return claims


def compact_context(text: str) -> str:
    s = text or ""
    # Total budget for all embeddings blocks combined
    total = 4800

    blocks = _parse_blocks(s)
    if not blocks:
        return s
    # Optional token mapping compression (default ON)
    map_on = True
    meta_body = ""
    if map_on:
        blocks2, meta_body = _token_map_compress(blocks)
        for k, v in blocks2.items():
            blocks[k] = v
        if meta_body:
            blocks["embeddings_meta"] = meta_body
    budgets = _budget_plan(blocks, total)
    # First compact each block, then cross-dedupe lines by preference order
    comp_map: Dict[str, str] = {}
    for tag, _label in _TAGS:
        if tag in blocks and blocks[tag].strip():
            body = blocks[tag]
            comp_map[tag] = _compact_one(tag, body, budgets.get(tag, len(body)))
    # Code range only mode (env-gated): replace code with compact range references and claims
    ranges_only = False
    extra_meta_lines: list[str] = []
    if ranges_only and (blocks.get("embeddings_code") or "").strip():
        comp_map, claim_lines = _code_ranges_only(blocks, comp_map)
        if claim_lines:
            extra_meta_lines.extend(claim_lines)
    # Cross-claims for refs/graph/memory summarizing token usage
    xclaims = _build_cross_claims(comp_map)
    if xclaims:
        extra_meta_lines.extend(xclaims)
    gclaims = _build_global_claims(comp_map)
    if gclaims:
        extra_meta_lines.extend(gclaims)
    wclaims = _build_weight_claims(comp_map)
    if wclaims:
        extra_meta_lines.extend(wclaims)
    # Relation claims from symbol index (requires P# path mapping in meta)
    rclaims = _build_relation_claims(blocks, comp_map)
    if rclaims:
        extra_meta_lines.extend(rclaims)
    # Cross-block line dedupe: keep code lines, drop duplicates later in refs/graph/memory/brain
    order = ["embeddings_code","embeddings_refs","embeddings_graph","embeddings_memory","embeddings_brain","embeddings_meta"]
    seen: set[str] = set()
    for tag in order:
        if tag not in comp_map:
            continue
        lines = (comp_map[tag] or "").splitlines()
        out_lines = []
        for ln in lines:
            key = ln.strip()
            if key and key in seen and tag != "embeddings_code":
                continue
            out_lines.append(ln)
            if key:
                seen.add(key)
        comp_map[tag] = "\n".join(out_lines)
    out = s
    for tag, _label in _TAGS:
        if tag in comp_map and (comp_map[tag] or "").strip():
            comp = comp_map[tag]
            out = _replace_once(out, tag, comp)
    # Append or create <embeddings_meta> with claims
    if extra_meta_lines:
        meta_prev = (blocks.get("embeddings_meta") or "").strip()
        meta_new = (meta_prev + ("\n" if meta_prev else "") + "\n".join(extra_meta_lines)).strip()
        if meta_new:
            if "<embeddings_meta>" in out:
                out = _replace_once(out, "embeddings_meta", meta_new)
            else:
                out = out + ("\n\n<embeddings_meta>\n" + meta_new + "\n</embeddings_meta>")
    return out


__all__ = ["compact_context"]
