from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.analysis_routes import router as analysis_router
from app.api.dashboard_routes import router as dashboard_router
from app.api.metrics_routes import router as metrics_router
from app.api.quality_routes import router as quality_router
from app.api.rag_routes import router as rag_router
from app.api.retrieval_routes import router as retrieval_router
from app.api.routes import router
from app.api.trace_routes import router as trace_router
from app.core.config import get_settings
from app.core.errors import DependencyUnavailableError


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(router)
    app.include_router(rag_router)
    app.include_router(retrieval_router)
    app.include_router(trace_router)
    app.include_router(metrics_router)
    app.include_router(analysis_router)
    app.include_router(quality_router)
    app.include_router(dashboard_router)

    @app.exception_handler(DependencyUnavailableError)
    async def dependency_unavailable_handler(
        request: Request,
        exc: DependencyUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    return app


app = create_app()
