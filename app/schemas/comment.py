from datetime import datetime

from pydantic import BaseModel, Field


class CommentBase(BaseModel):
    """
    CommentBase is used to define the data in the database
    """

    cohort_definition_id: int | None = Field(None, index=True)
    person_id: int = Field(..., index=True)
    comment: str
    datetime: datetime


class CommentCreate(CommentBase):
    """
    CommentCreate is used to create the data in the database
    """


class CommentRead(CommentBase):
    """
    CommentRead is used to read the data from the database
    """

    comment_id: int = Field(..., index=True)

    class Config:
        """
        orm_mode = True allows to read the data from the database
        """

        orm_mode = True
