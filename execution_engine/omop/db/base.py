# coding: utf-8
import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase

DateTimeWithTimeZone = DateTime(timezone=True)


class Base(DeclarativeBase):  # noqa: D101
    type_annotation_map = {
        datetime.datetime: DateTimeWithTimeZone,
    }


metadata = Base.metadata
