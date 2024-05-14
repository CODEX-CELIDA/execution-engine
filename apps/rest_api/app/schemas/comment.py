from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CommentBase(BaseModel):
    """
    CommentBase is used to define the data in the database
    """

    recommendation_id: int | None = Field(None)
    person_id: int = Field(...)
    text: str
    datetime: datetime


class CommentCreate(CommentBase):
    """
    CommentCreate is used to create the data in the database
    """


class CommentRead(CommentBase):
    """
    CommentRead is used to read the data from the database
    """

    comment_id: int = Field(...)
    model_config = ConfigDict(from_attributes=True)
