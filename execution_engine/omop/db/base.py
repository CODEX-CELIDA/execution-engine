# coding: utf-8
import datetime

from sqlalchemy import TIMESTAMP
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):  # noqa: D101
    type_annotation_map = {
        datetime.datetime: TIMESTAMP(timezone=True),
    }


metadata = Base.metadata
