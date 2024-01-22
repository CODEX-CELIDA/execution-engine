import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.app_state import AppState
from app.routers import comment, patient, recommendation

sys.path.append("..")

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Initialize the app state before the server starts and clean up after the server stops.
    """
    # Load all recommendations
    AppState.initialize()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> dict:
    """Server greeting"""
    return {"message": "CODEX-CELIDA Execution Engine"}


app.include_router(recommendation.router)
app.include_router(patient.router)
app.include_router(comment.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8001  # nosec (binding to all interfaces is desired)
    )
