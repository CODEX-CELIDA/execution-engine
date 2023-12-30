# coding: utf-8
import datetime

from sqlalchemy import TIMESTAMP, DateTime
from sqlalchemy.orm import DeclarativeBase

DateTimeWithTimeZone = DateTime(timezone=True)


class Base(DeclarativeBase):  # noqa: D101
    type_annotation_map = {
        datetime.datetime: TIMESTAMP(timezone=True),
    }


metadata = Base.metadata
