from __future__ import annotations

from fastapi import FastAPI
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks for Jinx runtime integration."""
    # Startup: connect to Jinx runtime
    try:
        from jinx.micro.runtime.api import ensure_runtime
        await ensure_runtime()
    except Exception:
        pass
    yield
    # Shutdown: cleanup
    try:
        from jinx.micro.runtime.api import stop_selfstudy
        await stop_selfstudy()
    except Exception:
        pass


app = FastAPI(
    title='Jinx API',
    version='1.0.0',
    description='Autonomous Engineering Agent API',
    lifespan=lifespan,
)

from .routers.emb import router as emb_router
from .routers.log import router as log_router
from .routers import jinx_router

app.include_router(emb_router, prefix='/emb', tags=['emb'])
app.include_router(log_router, prefix='/log', tags=['log'])
app.include_router(jinx_router, prefix='/jinx', tags=['jinx'])
