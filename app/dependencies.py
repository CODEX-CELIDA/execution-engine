import sqlalchemy.orm
from fastapi import HTTPException

from app.app_state import AppState
from execution_engine.clients import omopdb
from execution_engine.execution_engine import ExecutionEngine


def get_db() -> sqlalchemy.orm.Session:
    """
    Get database session
    """
    session = omopdb.session()
    try:
        yield session
    finally:
        session.close()


def get_execution_engine() -> ExecutionEngine:
    """
    Get execution engine
    """
    try:
        return AppState.get_execution_engine()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="ExecutionEngine not initialized"
        ) from e


def get_recommendations() -> dict:
    """
    Get available recommendations by URL
    """
    try:
        return AppState.get_recommendations()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Recommendations not initialized"
        ) from e
