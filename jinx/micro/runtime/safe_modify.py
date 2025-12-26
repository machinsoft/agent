"""Safe Self-Modification System - Code changes with automatic rollback.

Enables Jinx to modify its own code safely:
- Backup before any modification
- Syntax validation before applying
- Automatic rollback on failure
- Learning from modification outcomes
"""

from __future__ import annotations

import ast
import asyncio
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

_BACKUP_DIR = Path(".jinx") / "backups"
_MAX_BACKUPS = 50
_LOCK = Lock()


@dataclass
class Modification:
    """A tracked code modification."""
    id: str
    target_file: str
    original_content: str
    new_content: str
    backup_path: str
    timestamp: float = field(default_factory=time.time)
    applied: bool = False
    validated: bool = False
    rolled_back: bool = False
    error: Optional[str] = None


# Track recent modifications for rollback
_modifications: Dict[str, Modification] = {}


def _ensure_backup_dir() -> None:
    try:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _cleanup_old_backups() -> None:
    """Remove old backups to prevent disk bloat."""
    try:
        backups = sorted(_BACKUP_DIR.glob("*.bak"), key=lambda p: p.stat().st_mtime)
        if len(backups) > _MAX_BACKUPS:
            for old in backups[:-_MAX_BACKUPS]:
                old.unlink()
    except Exception:
        pass


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python code syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def validate_file_syntax(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate syntax of a file based on extension."""
    path = Path(file_path)
    
    if not path.exists():
        return False, "File does not exist"
    
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    if path.suffix == ".py":
        return validate_python_syntax(content)
    
    # For non-Python files, just check it's readable
    return True, None


def create_backup(file_path: str) -> Optional[str]:
    """Create a backup of a file before modification."""
    _ensure_backup_dir()
    
    path = Path(file_path)
    if not path.exists():
        return None
    
    try:
        timestamp = int(time.time() * 1000)
        backup_name = f"{path.stem}_{timestamp}{path.suffix}.bak"
        backup_path = _BACKUP_DIR / backup_name
        
        shutil.copy2(path, backup_path)
        _cleanup_old_backups()
        
        return str(backup_path)
    except Exception:
        return None


def restore_backup(backup_path: str, target_path: str) -> bool:
    """Restore a file from backup."""
    try:
        shutil.copy2(backup_path, target_path)
        return True
    except Exception:
        return False


async def safe_modify(
    file_path: str,
    old_code: str,
    new_code: str,
    validate: bool = True,
    require_syntax_valid: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """Safely modify a file with automatic backup and validation.
    
    Returns (success, message, modification_id).
    """
    path = Path(file_path)
    mod_id = f"mod_{int(time.time() * 1000)}"
    
    # Validate target file exists
    if not path.exists():
        return False, "Target file does not exist", None
    
    # Read current content
    try:
        current_content = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read file: {e}", None
    
    # Check if old_code exists in file
    if old_code and old_code not in current_content:
        return False, "Old code not found in file", None
    
    # Prepare new content
    if old_code:
        new_content = current_content.replace(old_code, new_code, 1)
    else:
        new_content = current_content + "\n" + new_code
    
    # Validate new content syntax
    if validate and require_syntax_valid and path.suffix == ".py":
        valid, error = validate_python_syntax(new_content)
        if not valid:
            return False, f"New code has syntax error: {error}", None
    
    # Create backup
    backup_path = create_backup(file_path)
    if not backup_path:
        return False, "Failed to create backup", None
    
    # Track modification
    mod = Modification(
        id=mod_id,
        target_file=file_path,
        original_content=current_content,
        new_content=new_content,
        backup_path=backup_path,
        validated=True,
    )
    
    with _LOCK:
        _modifications[mod_id] = mod
    
    # Apply modification
    try:
        path.write_text(new_content, encoding="utf-8")
        mod.applied = True
        
        # Verify file can still be parsed
        if validate and path.suffix == ".py":
            valid, error = validate_file_syntax(file_path)
            if not valid:
                # Rollback
                await rollback_modification(mod_id)
                return False, f"Validation failed after write: {error}", None
        
        # Record success for evolution
        try:
            from jinx.micro.runtime.self_evolution import learn
            learn(
                category="success_strategy",
                description=f"Safe modification succeeded",
                context=f"File: {file_path}",
                solution=f"Modified {len(old_code)} -> {len(new_code)} chars",
                confidence=0.7,
            )
        except Exception:
            pass
        
        return True, "Modification applied successfully", mod_id
        
    except Exception as e:
        # Attempt rollback
        if backup_path:
            restore_backup(backup_path, file_path)
        mod.error = str(e)
        return False, f"Failed to apply modification: {e}", None


async def rollback_modification(mod_id: str) -> Tuple[bool, str]:
    """Rollback a specific modification."""
    mod = _modifications.get(mod_id)
    if not mod:
        return False, "Modification not found"
    
    if mod.rolled_back:
        return True, "Already rolled back"
    
    try:
        # Restore from backup
        if restore_backup(mod.backup_path, mod.target_file):
            mod.rolled_back = True
            
            # Record rollback for evolution
            try:
                from jinx.micro.runtime.self_evolution import learn
                learn(
                    category="error_pattern",
                    description=f"Modification required rollback",
                    context=f"File: {mod.target_file}",
                    confidence=0.5,
                )
            except Exception:
                pass
            
            return True, "Rollback successful"
        else:
            return False, "Failed to restore backup"
            
    except Exception as e:
        return False, f"Rollback error: {e}"


async def rollback_all_recent(max_age_sec: float = 300) -> int:
    """Rollback all modifications within max_age_sec."""
    cutoff = time.time() - max_age_sec
    rolled_back = 0
    
    for mod_id, mod in list(_modifications.items()):
        if mod.timestamp > cutoff and mod.applied and not mod.rolled_back:
            success, _ = await rollback_modification(mod_id)
            if success:
                rolled_back += 1
    
    return rolled_back


def get_modification_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent modification history."""
    mods = sorted(_modifications.values(), key=lambda m: m.timestamp, reverse=True)
    return [
        {
            "id": m.id,
            "file": m.target_file,
            "timestamp": m.timestamp,
            "applied": m.applied,
            "rolled_back": m.rolled_back,
            "error": m.error,
        }
        for m in mods[:limit]
    ]


async def apply_evolution_proposal(proposal_id: str) -> Tuple[bool, str]:
    """Apply a code proposal from self_evolution with safe modification.
    
    Integrates with self_evolution.py proposals.
    """
    try:
        from jinx.micro.runtime.self_evolution import _proposals, CodeProposal
        
        proposal = _proposals.get(proposal_id)
        if not proposal:
            return False, "Proposal not found"
        
        if proposal.status != "pending":
            return False, f"Proposal status is {proposal.status}"
        
        # Apply with safe_modify
        success, message, mod_id = await safe_modify(
            file_path=proposal.target_file,
            old_code=proposal.old_code,
            new_code=proposal.new_code,
            validate=True,
            require_syntax_valid=True,
        )
        
        if success:
            proposal.status = "applied"
            proposal.applied_at = time.time()
            proposal.result = message
        else:
            proposal.status = "failed"
            proposal.result = message
        
        return success, message
        
    except Exception as e:
        return False, f"Error applying proposal: {e}"


__all__ = [
    "safe_modify",
    "rollback_modification",
    "rollback_all_recent",
    "validate_python_syntax",
    "validate_file_syntax",
    "create_backup",
    "restore_backup",
    "get_modification_history",
    "apply_evolution_proposal",
    "Modification",
]
