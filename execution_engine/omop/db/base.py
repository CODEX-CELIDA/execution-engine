# coding: utf-8

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):  # noqa: D101
    pass


metadata = Base.metadata
