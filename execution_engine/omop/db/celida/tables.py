from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Connection,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Table,
    event,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.base import Base
from execution_engine.omop.db.celida.schema import SCHEMA_NAME
from execution_engine.omop.db.celida.triggers import (
    create_trigger_interval_overlap_check_sql,
    trigger_interval_overlap_check_function_sql,
)
from execution_engine.omop.db.omop.schema import SCHEMA_NAME as OMOP_SCHEMA_NAME
from execution_engine.util.interval import IntervalType

IntervalTypeEnum = Enum(IntervalType, name="interval_type", schema=SCHEMA_NAME)
CohortCategoryEnum = Enum(CohortCategory, name="cohort_category", schema=SCHEMA_NAME)


class Recommendation(Base):  # noqa: D101
    __tablename__ = "recommendation"
    __table_args__ = {"schema": SCHEMA_NAME}

    recommendation_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_name: Mapped[str]
    recommendation_title: Mapped[str]
    recommendation_url: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    recommendation_version: Mapped[str]
    recommendation_package_version: Mapped[str]
    recommendation_hash: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )
    recommendation_json = mapped_column(LargeBinary)
    create_datetime: Mapped[datetime]


class PopulationInterventionPair(Base):  # noqa: D101
    __tablename__ = "population_intervention_pair"
    __table_args__ = {"schema": SCHEMA_NAME}

    pi_pair_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
    )
    pi_pair_url: Mapped[str]
    pi_pair_name: Mapped[str]
    pi_pair_hash: Mapped[str] = mapped_column(String(64), index=True, unique=True)


class Criterion(Base):  # noqa: D101
    __tablename__ = "criterion"
    __table_args__ = {"schema": SCHEMA_NAME}

    criterion_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    # todo: add link to recommendation or 1:n to population/intervention pair?
    criterion_description: Mapped[str]
    criterion_hash: Mapped[str] = mapped_column(String(64), index=True, unique=True)


class ExecutionRun(Base):  # noqa: D101
    __tablename__ = "execution_run"
    __table_args__ = {"schema": SCHEMA_NAME}

    run_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_id = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
    )
    observation_start_datetime: Mapped[datetime]
    observation_end_datetime: Mapped[datetime]
    run_datetime: Mapped[datetime]
    engine_version: Mapped[str]

    recommendation: Mapped["Recommendation"] = relationship(
        primaryjoin="ExecutionRun.recommendation_id == Recommendation.recommendation_id",
    )


class ResultInterval(Base):  # noqa: D101
    __tablename__ = "result_interval"
    __table_args__ = (
        Index(
            "ix_rec_result_int_run_id_cohort_category_person_id_valid_date",
            "run_id",
            "cohort_category",
            "person_id",
            "interval_start",
            "interval_end",
        ),
        Index(
            "ix_rec_result_int_run_id_pi_pair_id_criterion_id_valid_date",
            "run_id",
            "pi_pair_id",
            "criterion_id",
            "person_id",
            "interval_start",
            "interval_end",
        ),
        Index(
            "ix_rec_result_int_category_run_id_person_id",
            "cohort_category",
            "run_id",
            "person_id",
        ),
        {"schema": SCHEMA_NAME},
    )

    result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    run_id = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.execution_run.run_id"),
        index=True,
    )
    pi_pair_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.population_intervention_pair.pi_pair_id"),
        index=True,
        nullable=True,
    )
    criterion_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.criterion.criterion_id"),
        index=True,
        nullable=True,
    )
    cohort_category = mapped_column(CohortCategoryEnum)
    person_id: Mapped[int] = mapped_column(
        BigInteger(), ForeignKey(f"{OMOP_SCHEMA_NAME}.person.person_id"), index=True
    )
    interval_start: Mapped[datetime]
    interval_end: Mapped[datetime]
    interval_type = mapped_column(IntervalTypeEnum)

    execution_run: Mapped["ExecutionRun"] = relationship(
        primaryjoin="ResultInterval.run_id == ExecutionRun.run_id",
    )

    population_intervention_pair: Mapped["PopulationInterventionPair"] = relationship(
        primaryjoin="ResultInterval.pi_pair_id == PopulationInterventionPair.pi_pair_id",
    )

    criterion: Mapped["Criterion"] = relationship(
        primaryjoin="ResultInterval.criterion_id == Criterion.criterion_id",
    )


@event.listens_for(ResultInterval.__table__, "after_create")
def create_interval_overlap_check_triggers(
    target: Table, connection: Connection, **kw: Any
) -> None:
    """
    Create triggers for the result_interval table.
    """
    connection.execute(
        text(
            trigger_interval_overlap_check_function_sql.format(
                schema=target.schema, table=target.name
            )
        )
    )
    connection.execute(
        text(
            create_trigger_interval_overlap_check_sql.format(
                schema=target.schema, table=target.name
            )
        )
    )


class Comment(Base):  # noqa: D101
    __tablename__ = "comment"
    __table_args__ = {"schema": SCHEMA_NAME}

    comment_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
        nullable=True,
    )

    person_id: Mapped[int] = mapped_column(
        BigInteger(), ForeignKey(f"{OMOP_SCHEMA_NAME}.person.person_id"), index=True
    )

    text: Mapped[str]
    datetime: Mapped[datetime]
