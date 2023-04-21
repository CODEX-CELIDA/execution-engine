import pytest

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)


@pytest.mark.recommendation
class TestRecommendation36aPeep:
    pass
