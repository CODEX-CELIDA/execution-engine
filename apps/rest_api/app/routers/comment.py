from app.dependencies import get_db
from app.schemas.comment import CommentCreate, CommentRead
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from execution_engine.omop.db.celida.tables import Comment

router = APIRouter()


@router.get("/comments/{person_id}", response_model=list[CommentRead])
def get_comments(person_id: int, db: Session = Depends(get_db)) -> list[CommentRead]:
    """
    Get all comments for a person.
    """
    comments = db.query(Comment).filter(Comment.person_id == person_id).all()
    if not comments:
        raise HTTPException(status_code=404, detail="Comments not found")
    return comments


@router.post("/comments/", response_model=CommentRead)
def create_comment(comment: CommentCreate, db: Session = Depends(get_db)) -> Comment:
    """
    Create a new comment in the database.
    """
    db_comment = Comment(**comment.dict())
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)

    return db_comment
