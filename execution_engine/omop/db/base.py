# coding: utf-8
from datetime import datetime

import sqlalchemy
from sqlalchemy import TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.declarative import declarative_base
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


Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata
