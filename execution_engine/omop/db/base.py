# coding: utf-8

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata
