# coding: utf-8
from datetime import date, datetime

import sqlalchemy
from sqlalchemy import TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta


class DateTime(TypeDecorator):
    """
    SQLAlchemy type for datetime columns.
    """

    impl = sqlalchemy.types.DateTime
    cache_ok = True

    def process_literal_param(self, value: datetime, dialect: Dialect) -> str:
        """
        Convert a datetime to a string.
        """
        return value.strftime("'%Y-%m-%d %H:%M:%S'")


class Date(TypeDecorator):
    """
    SQLAlchemy type for date columns.
    """

    impl = sqlalchemy.types.Date
    cache_ok = True

    def process_literal_param(self, value: date, dialect: Dialect) -> str:
        """
        Convert a date to a string.
        """
        return value.strftime("'%Y-%m-%d'")


Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata
