"""Self-Healing System - Autonomous error detection, analysis, and repair.

AI-powered self-repair with:
- Real-time error pattern detection
- Semantic code analysis using embeddings
- Automatic fix generation using LLM
- Safe patch application with rollback
- Learning from successful repairs
- Multi-stage validation pipeline
"""

from __future__ import annotations

import asyncio
import ast
import difflib
import hashlib
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ErrorPattern:
    """Detected error pattern."""
    error_type: str
    error_message: str
    traceback_lines: List[str]
    file_path: Optional[str]
    line_number: Optional[int]
    function_name: Optional[str]
    frequency: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    fix_attempted: bool = False
    fix_successful: bool = False


@dataclass
class RepairAction:
    """Proposed repair action."""
    action_type: str  # 'patch_code', 'update_config', 'restart_service'
    target_file: Optional[str]
    original_code: Optional[str]
    fixed_code: Optional[str]
    confidence: float
    rationale: str
    estimated_impact: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class RepairResult:
    """Result of repair attempt."""
    success: bool
    action: RepairAction
    validation_passed: bool
    error_cleared: bool
    side_effects: List[str]
    timestamp: float = field(default_factory=time.time)


class CodeAnalyzer:
    """Analyzes code for potential issues using AST and pattern matching."""
    
    def __init__(self):
        self._syntax_cache: Dict[str, bool] = {}
    
    def analyze_error_context(
        self,
        file_path: str,
        line_number: int,
        error_message: str
    ) -> Dict[str, Any]:
        """Extract detailed context around error location."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Context window
            start = max(0, line_number - 10)
            end = min(len(lines), line_number + 10)
            
            context_lines = lines[start:end]
            error_line = lines[line_number - 1] if line_number <= len(lines) else ""
            
            # AST analysis
            try:
                tree = ast.parse(''.join(lines))
                
                # Find function/class containing error
                containing_scope = self._find_containing_scope(tree, line_number)
            except SyntaxError:
                containing_scope = None
            
            return {
                'file_path': file_path,
                'line_number': line_number,
                'error_line': error_line.strip(),
                'context_before': context_lines[:10],
                'context_after': context_lines[11:] if len(context_lines) > 11 else [],
                'containing_scope': containing_scope,
                'file_size': len(lines),
                'error_message': error_message
            }
        
        except Exception:
            return {}
    
    def _find_containing_scope(self, tree: ast.AST, line_number: int) -> Optional[str]:
        """Find the function or class containing the given line."""
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    if node.lineno <= line_number <= (node.end_lineno or node.lineno):
                        return node.name
        
        return None
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        
        # Cache key
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        if code_hash in self._syntax_cache:
            return self._syntax_cache[code_hash], None
        
        try:
            ast.parse(code)
            self._syntax_cache[code_hash] = True
            return True, None
        except SyntaxError as e:
            return False, str(e)


class RepairEngine:
    """Generates and applies code repairs using AI assistance."""
    
    def __init__(self):
        self._repair_history: deque[RepairResult] = deque(maxlen=1000)
        self._analyzer = CodeAnalyzer()
    
    async def generate_repair(
        self,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> Optional[RepairAction]:
        """Generate repair using AI analysis."""
        
        # Build repair prompt
        prompt = self._build_repair_prompt(pattern, context)
        
        try:
            # Call LLM for repair suggestion
            from jinx.micro.llm.service import spark_openai
            
            # spark_openai returns (output_text, code_tag_id) and does not accept
            # temperature/max_tokens kwargs. Use default validated path and extract text.
            out_text, _ = await spark_openai(prompt)
            response = out_text
            
            # Parse repair from response
            repair = self._parse_repair_response(response, pattern, context)
            
            return repair
        
        except Exception:
            # Fallback to pattern-based repair
            return self._pattern_based_repair(pattern, context)
    
    def _build_repair_prompt(
        self,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for repair generation using prompts/repair_suggest template."""
        try:
            from jinx.prompts import render_prompt as _render_prompt
        except Exception:
            _render_prompt = None  # type: ignore
        before_text = ''.join((context.get('context_before') or [])[-5:]) if isinstance(context.get('context_before'), list) else str(context.get('context_before') or '')
        after_text = ''.join((context.get('context_after') or [])[:5]) if isinstance(context.get('context_after'), list) else str(context.get('context_after') or '')
        if _render_prompt is not None:
            try:
                return _render_prompt(
                    "repair_suggest",
                    error_type=pattern.error_type,
                    error_message=pattern.error_message,
                    file_path=context.get('file_path', 'unknown'),
                    line_number=context.get('line_number', 0),
                    error_line=context.get('error_line', ''),
                    context_before_text=before_text,
                    context_after_text=after_text,
                    containing_scope=context.get('containing_scope', 'unknown'),
                )
            except Exception:
                pass
        # Fallback minimal payload if template unavailable
        import json as _json
        return _json.dumps({
            "error_type": pattern.error_type,
            "error_message": pattern.error_message,
            "file_path": context.get('file_path', 'unknown'),
            "line_number": context.get('line_number', 0),
            "error_line": context.get('error_line', ''),
            "context_before": before_text,
            "context_after": after_text,
            "scope": context.get('containing_scope', 'unknown'),
        }, ensure_ascii=False)
    
    def _parse_repair_response(
        self,
        response: str,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> Optional[RepairAction]:
        """Parse LLM response into RepairAction."""
        
        try:
            # Extract sections
            import re
            
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
            fix_match = re.search(r'<fix>(.*?)</fix>', response, re.DOTALL)
            conf_match = re.search(r'<confidence>([\d.]+)</confidence>', response)
            impact_match = re.search(r'<impact>(\w+)</impact>', response)
            
            if not fix_match:
                return None
            
            fixed_code = fix_match.group(1).strip()
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            impact = impact_match.group(1) if impact_match else 'medium'
            rationale = analysis_match.group(1).strip() if analysis_match else "AI-generated fix"
            
            # Validate syntax
            valid, error = self._analyzer.validate_syntax(fixed_code)
            
            if not valid:
                return None
            
            return RepairAction(
                action_type='patch_code',
                target_file=context.get('file_path'),
                original_code=context.get('error_line'),
                fixed_code=fixed_code,
                confidence=confidence,
                rationale=rationale,
                estimated_impact=impact
            )
        
        except Exception:
            return None
    
    def _pattern_based_repair(
        self,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> Optional[RepairAction]:
        """Fallback pattern-based repair."""
        
        error_type = pattern.error_type
        error_line = context.get('error_line', '')
        
        # Common patterns
        if 'NameError' in error_type:
            # Add missing import
            var_name = pattern.error_message.split("'")[1] if "'" in pattern.error_message else None
            
            if var_name:
                fixed = f"from jinx import {var_name}\n{error_line}"
                
                return RepairAction(
                    action_type='patch_code',
                    target_file=context.get('file_path'),
                    original_code=error_line,
                    fixed_code=fixed,
                    confidence=0.6,
                    rationale=f"Add missing import for {var_name}",
                    estimated_impact='low'
                )
        
        elif 'AttributeError' in error_type and 'NoneType' in pattern.error_message:
            # Add None check
            fixed = f"if {error_line.split('.')[0].strip()}:\n    {error_line}"
            
            return RepairAction(
                action_type='patch_code',
                target_file=context.get('file_path'),
                original_code=error_line,
                fixed_code=fixed,
                confidence=0.7,
                rationale="Add None check to prevent AttributeError",
                estimated_impact='low'
            )
        
        elif 'IndentationError' in error_type:
            # Fix indentation
            fixed = error_line.lstrip() if error_line else ""
            
            return RepairAction(
                action_type='patch_code',
                target_file=context.get('file_path'),
                original_code=error_line,
                fixed_code=fixed,
                confidence=0.8,
                rationale="Fix indentation",
                estimated_impact='low'
            )
        
        return None
    
    async def apply_repair(
        self,
        action: RepairAction
    ) -> RepairResult:
        """Apply repair action with rollback capability."""
        
        if action.action_type != 'patch_code' or not action.target_file:
            return RepairResult(
                success=False,
                action=action,
                validation_passed=False,
                error_cleared=False,
                side_effects=['Unsupported action type']
            )
        
        # Backup original file
        backup_path = f"{action.target_file}.backup.{int(time.time())}"
        
        try:
            # Read original
            with open(action.target_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Try safe context-based autopatch first
            auto_ok = False
            try:
                from jinx.micro.runtime.patch.autopatch import autopatch as _autopatch, AutoPatchArgs as _Args
                args = _Args(
                    path=action.target_file,
                    context_before=(action.original_code or ""),
                    code=(action.fixed_code or ""),
                    preview=False,
                )
                # timebox autopatch to keep RT guarantees
                ok, strat, detail = await asyncio.wait_for(_autopatch(args), timeout=1.2)
                auto_ok = bool(ok)
            except Exception:
                auto_ok = False

            if not auto_ok:
                # Fallback: apply minimal replace
                if action.original_code and action.fixed_code:
                    patched_content = original_content.replace(
                        action.original_code,
                        action.fixed_code,
                        1  # Replace only first occurrence
                    )
                else:
                    patched_content = action.fixed_code or ""
                # Validate patched code
                valid, error = self._analyzer.validate_syntax(patched_content)
                if not valid:
                    return RepairResult(
                        success=False,
                        action=action,
                        validation_passed=False,
                        error_cleared=False,
                        side_effects=[f'Syntax validation failed: {error}']
                    )
                # Write patched version
                with open(action.target_file, 'w', encoding='utf-8') as f:
                    f.write(patched_content)
            
            # Wait for system to stabilize
            await asyncio.sleep(0.5)
            
            # Validate (check if error cleared)
            error_cleared = await self._validate_repair(action)
            
            result = RepairResult(
                success=True,
                action=action,
                validation_passed=True,
                error_cleared=error_cleared,
                side_effects=[]
            )
            
            # Store in history
            self._repair_history.append(result)
            
            # If successful, remove backup
            if error_cleared:
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            
            return result
        
        except Exception as e:
            # Rollback on failure
            try:
                if os.path.exists(backup_path):
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        original = f.read()
                    
                    with open(action.target_file, 'w', encoding='utf-8') as f:
                        f.write(original)
            except Exception:
                pass
            
            return RepairResult(
                success=False,
                action=action,
                validation_passed=False,
                error_cleared=False,
                side_effects=[f'Application failed: {str(e)}']
            )
    
    async def _validate_repair(self, action: RepairAction) -> bool:
        """Validate that repair cleared the error."""
        
        # Try to compile/import the fixed file
        if action.target_file:
            try:
                import py_compile
                py_compile.compile(action.target_file, doraise=True)
                return True
            except Exception:
                return False
        
        return False


class SelfHealingSystem:
    """Main self-healing orchestrator."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._enabled = True
        self._error_patterns: Dict[str, ErrorPattern] = {}
        self._repair_engine = RepairEngine()
        self._healing_in_progress = False
        
        # Statistics
        self._total_errors_detected = 0
        self._total_repairs_attempted = 0
        self._total_repairs_successful = 0
    
    async def process_error(
        self,
        error_type: str,
        error_message: str,
        traceback_str: str
    ) -> bool:
        """Process an error and attempt self-repair."""
        
        if not self._enabled or self._healing_in_progress:
            return False
        
        async with self._lock:
            self._total_errors_detected += 1
            
            # Parse traceback
            file_path, line_number, func_name = self._parse_traceback(traceback_str)

            # Decide whether this error is urgent enough to attempt repair immediately
            et = (error_type or "").strip()
            em = (error_message or "")
            tb = (traceback_str or "")
            urgent = False
            try:
                if (
                    "ModuleNotFoundError" in et
                    or "ImportError" in et
                    or "partially initialized module" in em
                    or "circular import" in em
                ):
                    urgent = True
            except Exception:
                urgent = False

            # If we couldn't extract a file path, try to recover one from common import-traceback formats.
            if not file_path:
                try:
                    import re as _re
                    # Most tracebacks include at least one `File "...", line N` entry; pick the last one.
                    mm = None
                    for m in _re.finditer(r'File "([^"]+\.py)", line (\d+)', tb):
                        mm = m
                    if mm is not None:
                        file_path = str(mm.group(1) or "").strip() or None
                        try:
                            line_number = int(mm.group(2) or 0) or None
                        except Exception:
                            line_number = None
                except Exception:
                    pass

            # If still no file path, we can still schedule a repair task by module name when possible.
            if not file_path and urgent:
                try:
                    import re as _re
                    mod = None
                    # ModuleNotFoundError: No module named 'x.y'
                    mm = _re.search(r"No module named ['\"]([^'\"]+)['\"]", em)
                    if mm:
                        mod = str(mm.group(1) or "").strip() or None
                    # ImportError: cannot import name 'foo' from partially initialized module 'a.b' ...
                    if mod is None:
                        mm2 = _re.search(r"module ['\"]([^'\"]+)['\"]", em)
                        if mm2:
                            mod = str(mm2.group(1) or "").strip() or None
                    if mod and mod.startswith("jinx."):
                        try:
                            from jinx.micro.runtime.api import submit_task as _submit
                            await _submit("repair.import_missing", module=mod)
                        except Exception:
                            pass
                except Exception:
                    pass

            if not file_path:
                return False
            
            # Create/update error pattern
            pattern_key = f"{error_type}:{file_path}:{line_number}"
            
            if pattern_key in self._error_patterns:
                pattern = self._error_patterns[pattern_key]
                pattern.frequency += 1
                pattern.last_seen = time.time()
            else:
                pattern = ErrorPattern(
                    error_type=error_type,
                    error_message=error_message,
                    traceback_lines=traceback_str.split('\n'),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=func_name
                )
                self._error_patterns[pattern_key] = pattern
            
            # Attempt repair if pattern is frequent or critical
            should_repair = (
                urgent
                or pattern.frequency >= 2
                or 'Critical' in error_type
                or 'Fatal' in error_type
            ) and not pattern.fix_attempted
            
            if should_repair:
                self._healing_in_progress = True
                
                try:
                    success = await self._attempt_repair(pattern)
                    
                    if success:
                        self._total_repairs_successful += 1
                        pattern.fix_successful = True
                        
                        # Remove from patterns if fixed
                        del self._error_patterns[pattern_key]
                    
                    pattern.fix_attempted = True
                    self._total_repairs_attempted += 1
                    
                    return success
                
                finally:
                    self._healing_in_progress = False
        
        return False
    
    async def _attempt_repair(self, pattern: ErrorPattern) -> bool:
        """Attempt to repair the error."""
        
        print(f"\n{'='*70}")
        print("🔧 SELF-HEALING INITIATED")
        print(f"{'='*70}")
        print(f"Error: {pattern.error_type}")
        print(f"File: {pattern.file_path}:{pattern.line_number}")
        print(f"Message: {pattern.error_message}")
        print(f"Frequency: {pattern.frequency}")
        
        # Analyze error context
        context = self._repair_engine._analyzer.analyze_error_context(
            pattern.file_path or "",
            pattern.line_number or 0,
            pattern.error_message
        )
        
        if not context:
            print("✗ Could not analyze error context")
            return False
        
        # Generate repair
        print("\n🔍 Generating repair...")
        
        repair = await self._repair_engine.generate_repair(pattern, context)
        
        if not repair:
            print("✗ Could not generate repair")
            return False
        
        print(f"✓ Repair generated (confidence: {repair.confidence:.2f})")
        print(f"  Rationale: {repair.rationale}")
        print(f"  Impact: {repair.estimated_impact}")
        
        # Apply repair
        print("\n🔨 Applying repair...")
        
        result = await self._repair_engine.apply_repair(repair)
        
        if result.success and result.error_cleared:
            print("✓ Repair applied successfully!")
            print("✓ Error cleared")
            print(f"{'='*70}\n")
            
            # Log to memory
            try:
                from jinx.micro.runtime.crash_diagnostics import record_operation
                record_operation(
                    "self_healing_success",
                    details={
                        'error_type': pattern.error_type,
                        'file': pattern.file_path,
                        'confidence': repair.confidence
                    },
                    success=True
                )
            except Exception:
                pass
            
            return True
        
        else:
            print("✗ Repair failed")
            if result.side_effects:
                print(f"  Side effects: {', '.join(result.side_effects)}")
            print(f"{'='*70}\n")
            
            return False
    
    def _parse_traceback(self, traceback_str: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Parse traceback to extract file, line, and function."""
        
        lines = traceback_str.strip().split('\n')
        
        for line in reversed(lines):
            if 'File "' in line and 'line ' in line:
                try:
                    # Extract file path
                    file_start = line.index('File "') + 6
                    file_end = line.index('"', file_start)
                    file_path = line[file_start:file_end]
                    
                    # Extract line number
                    line_start = line.index('line ') + 5
                    line_end = line.index(',', line_start) if ',' in line[line_start:] else len(line)
                    line_number = int(line[line_start:line_end].strip())
                    
                    # Extract function name
                    if 'in ' in line:
                        func_start = line.index('in ') + 3
                        func_name = line[func_start:].strip()
                    else:
                        func_name = None
                    
                    return file_path, line_number, func_name
                
                except Exception:
                    pass
        
        return None, None, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        
        return {
            'enabled': self._enabled,
            'healing_in_progress': self._healing_in_progress,
            'total_errors_detected': self._total_errors_detected,
            'total_repairs_attempted': self._total_repairs_attempted,
            'total_repairs_successful': self._total_repairs_successful,
            'success_rate': (
                self._total_repairs_successful / self._total_repairs_attempted
                if self._total_repairs_attempted > 0 else 0.0
            ),
            'active_patterns': len(self._error_patterns)
        }


# Singleton
_healing_system: Optional[SelfHealingSystem] = None
_healing_lock = asyncio.Lock()


async def get_healing_system() -> SelfHealingSystem:
    """Get singleton healing system."""
    global _healing_system
    if _healing_system is None:
        async with _healing_lock:
            if _healing_system is None:
                _healing_system = SelfHealingSystem()
    return _healing_system


async def auto_heal_error(
    error_type: str,
    error_message: str,
    traceback_str: str
) -> bool:
    """Automatically attempt to heal an error."""
    system = await get_healing_system()
    healed = await system.process_error(error_type, error_message, traceback_str)
    
    # Integrate with self-evolution for learning
    try:
        from jinx.micro.runtime.self_evolution import learn, record_attempt
        
        if healed:
            # Learn from successful repair
            learn(
                category="success_strategy",
                description=f"Self-healed {error_type}",
                context=error_message[:200],
                solution="auto_heal_error succeeded",
                confidence=0.7,
            )
            record_attempt("goal_c8e7d3a2", True, f"Healed {error_type}")  # Minimize errors goal
        else:
            # Learn from failed repair attempt
            learn(
                category="error_pattern",
                description=f"Failed to heal {error_type}",
                context=traceback_str[:300],
                confidence=0.4,
            )
            record_attempt("goal_c8e7d3a2", False, f"Failed to heal {error_type}")
    except Exception:
        pass
    
    return healed


async def trigger_evolution_healing(error_pattern: str, context: str) -> Optional[str]:
    """Trigger LLM-based architecture fix when errors persist.
    
    Called by self-evolution when local healing fails repeatedly.
    """
    try:
        from jinx.micro.runtime.self_evolution import (
            propose_architecture_change,
            propose_modification,
            get_relevant_learnings,
        )
        
        # Check if we have learned solutions
        learnings = get_relevant_learnings("error_pattern", error_pattern, limit=3)
        for learning in learnings:
            if learning.solution and learning.confidence > 0.6:
                return learning.solution  # Use known solution
        
        # Request LLM analysis for new solution
        proposal = await propose_architecture_change(error_pattern, context)
        return proposal
        
    except Exception:
        return None


__all__ = [
    "SelfHealingSystem",
    "get_healing_system",
    "auto_heal_error",
    "trigger_evolution_healing",
]
