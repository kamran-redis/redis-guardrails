from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from redis_guardrails.cli.core import DEFAULT_MODEL, DEFAULT_REDIS_URL, build_service
from redis_guardrails.web.errors import register_exception_handlers
from redis_guardrails.web.routes import home

STATIC_DIR = Path(__file__).parent / "static"


def _build_app() -> FastAPI:
    """Wires routers/static only — no service, no lifespan. Shared by
    create_app() (production) and tests, so route registration never
    duplicates between the two."""
    app = FastAPI(title="redis_guardrails")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_exception_handlers(app)
    app.include_router(home.router)
    return app


def create_app(
    redis_url: str | None = None,
    model: str | None = None,
    overwrite: bool = False,
) -> FastAPI:
    redis_url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    model = model or os.environ.get("REDIS_GUARDRAILS_MODEL", DEFAULT_MODEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = build_service(redis_url=redis_url, model=model, overwrite=overwrite)
        yield

    app = _build_app()
    app.router.lifespan_context = lifespan
    return app
