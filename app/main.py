import sys

from app.app_state import AppState
from app.routers import comment, patient, recommendation

sys.path.append("..")

from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    """
    Load all recommendations
    Returns: None
    """
    AppState.initialize()


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
