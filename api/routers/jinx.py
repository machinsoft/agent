from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


router = APIRouter()


class TaskRequest(BaseModel):
    name: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    task_id: str
    status: str


class PromptRequest(BaseModel):
    text: str
    has_code_task: bool = True
    has_complex_reasoning: bool = False
    has_embeddings: bool = False
    is_error_recovery: bool = False


class PromptResponse(BaseModel):
    prompt: str
    length: int
    modules_used: List[str]


class ModulesResponse(BaseModel):
    modules: List[str]
    count: int


@router.get('/health')
async def health() -> dict:
    """Health check endpoint."""
    return {'status': 'ok', 'service': 'jinx'}


@router.get('/modules')
async def list_modules() -> ModulesResponse:
    """List available prompt modules."""
    try:
        from jinx.prompts.burning_logic import list_modules
        mods = list_modules()
        return ModulesResponse(modules=mods, count=len(mods))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/prompt')
async def build_prompt(req: PromptRequest) -> PromptResponse:
    """Build modular prompt based on task requirements."""
    try:
        from jinx.prompts.burning_logic import build_modular_prompt
        prompt = build_modular_prompt(
            has_code_task=req.has_code_task,
            has_complex_reasoning=req.has_complex_reasoning,
            has_embeddings=req.has_embeddings,
            is_error_recovery=req.is_error_recovery,
        )
        modules = []
        if req.has_code_task:
            modules.append('code')
        if req.has_complex_reasoning:
            modules.append('agents')
        if req.has_embeddings:
            modules.append('embeddings')
        if req.is_error_recovery:
            modules.append('error_recovery')
        return PromptResponse(
            prompt=prompt,
            length=len(prompt),
            modules_used=['identity', 'format', 'pulse'] + modules,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/task')
async def submit_task(req: TaskRequest) -> TaskResponse:
    """Submit a task to Jinx runtime."""
    try:
        from jinx.micro.runtime.api import submit_task as _submit
        tid = await _submit(req.name, *req.args, **req.kwargs)
        return TaskResponse(task_id=tid, status='submitted')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/programs')
async def list_programs() -> dict:
    """List running micro-programs."""
    try:
        from jinx.micro.runtime.api import list_programs as _list
        progs = await _list()
        return {'programs': progs, 'count': len(progs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
