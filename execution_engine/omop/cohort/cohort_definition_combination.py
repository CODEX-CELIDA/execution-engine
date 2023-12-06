import re
from typing import Any, Dict, Iterator, Self

from sqlalchemy import (
    Column,
    Date,
    Index,
    Insert,
    Integer,
    MetaData,
    Select,
    Table,
    and_,
    bindparam,
    select,
    union,
)
from sqlalchemy.orm import Query
from sqlalchemy.sql.ddl import DropTable

from execution_engine.constants import CohortCategory
from execution_engine.omop.cohort import add_result_insert
from execution_engine.omop.cohort.cohort_definition import CohortDefinition
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.db.celida import RecommendationResult
from execution_engine.omop.serializable import Serializable
from execution_engine.util.sql import SelectInto


class CohortDefinitionCombination(Serializable):
    """
    A cohort definition combination in OMOP as a collection of separate cohort definitions.

    A cohort definition represents an individual recommendation plan (i.e. one part of a single recommendation),
    whereas a cohort definition combination represents the whole recommendation, consisting of one or multiple
    recommendation plans = cohort definitions.
    """

    base_table: Table | None = None

    def __init__(
        self,
        cohort_definitions: list[CohortDefinition],
        base_criterion: Criterion,
        name: str,
        title: str,
        url: str,
        version: str,
        description: str,
        cohort_definition_id: int | None = None,
    ) -> None:
        self._cohort_definitions: list[CohortDefinition] = cohort_definitions
        self._base_criterion: Criterion = base_criterion
        self._name: str = name
        self._title: str = title
        self._url: str = url
        self._version: str = version
        self._description: str = description
        # The id is used in the cohort_definition_id field in the result tables.
        self._id = cohort_definition_id

    @property
    def name(self) -> str:
        """
        Get the name of the recommendation.
        """
        return self._name

    @property
    def title(self) -> str:
        """
        Get the title of the recommendation.
        """
        return self._title

    @property
    def url(self) -> str:
        """
        Get the url of the cohort definition combination.
        """
        return self._url

    @property
    def version(self) -> str:
        """
        Get the version of the cohort definition combination.
        """
        return self._version

    @property
    def description(self) -> str:
        """
        Get the description of the recommendation.
        """
        return self._description

    def criteria(self) -> CriterionCombination:
        """
        Get the criteria of the cohort definition combination.
        """
        criteria = CriterionCombination(
            name="root",
            exclude=False,
            category=CohortCategory.BASE,
            operator=CriterionCombination.Operator("OR"),
        )

        for cd in self._cohort_definitions:
            criteria.add(cd.criteria())

        return criteria

    def __iter__(self) -> Iterator[CohortDefinition]:
        """
        Iterate over the cohort definitions.
        """
        yield from self._cohort_definitions

    def __len__(self) -> int:
        """
        Get the number of cohort definitions.
        """
        return len(self._cohort_definitions)

    def __getitem__(self, index: int) -> CohortDefinition:
        """
        Get the cohort definition at the given index.
        """
        return self._cohort_definitions[index]

    @staticmethod
    def to_table(name: str) -> Table:
        """
        Convert a name to a valid SQL table name.
        """
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        metadata = MetaData()
        return Table(
            name,
            metadata,
            Column("person_id", Integer, primary_key=True),
            Column("valid_date", Date),
        )

    def create_base_table(self) -> Table:
        """
        Create the base table for the cohort definition combination.
        """
        self.base_table = self.to_table(self._base_criterion.name)

        query = self._base_criterion.sql_generate(base_table=self.base_table)
        query = self._base_criterion.sql_insert_into_table(
            query, self.base_table, temporary=True
        )
        assert isinstance(
            query, SelectInto
        ), f"Invalid query - expected SelectInto, got {type(query)}"

        yield query
        yield Index(
            "ix_person_id_valid_date",
            self.base_table.c.person_id,
            self.base_table.c.valid_date,
        )

        # let's also insert the base criterion into the recommendation_result table (for the full list of patients)
        query = add_result_insert(
            self.base_table.select(),
            plan_id=None,
            criterion_id=None,
            cohort_category=CohortCategory.BASE,
        )
        query.description = "Insert base criterion into recommendation_result table"
        yield query

    def process(self) -> Iterator[Query]:
        """
        Process the cohort definition combination into SQL statements.
        """

        yield from self.create_base_table()

        for i, cohort_definition in enumerate(self._cohort_definitions):
            yield from cohort_definition.process(
                base_table=self.base_table,
            )

        yield from self.combine()

    def combine(self) -> Iterator[Insert]:
        """
        Combine the results of the individual cohort definitions.
        """

        table = RecommendationResult.__table__

        def _get_query(
            cohort_definition: CohortDefinition, category: CohortCategory
        ) -> Select:
            return (
                select(table.c.person_id, table.c.valid_date)
                .distinct(table.c.person_id, table.c.valid_date)
                .where(
                    and_(
                        table.c.plan_id == cohort_definition.id,
                        table.c.cohort_category == category.name,
                        table.c.criterion_id.is_(None),
                        table.c.recommendation_run_id == bindparam("run_id"),
                    )
                )
            )

        def get_statements(cohort_category: CohortCategory) -> list[Select]:
            return [_get_query(cd, cohort_category) for cd in self._cohort_definitions]

        for category in [
            CohortCategory.POPULATION,
            CohortCategory.INTERVENTION,
            CohortCategory.POPULATION_INTERVENTION,
        ]:
            statements = get_statements(category)
            query = union(*statements)
            query = add_result_insert(
                query,
                plan_id=None,
                criterion_id=None,
                cohort_category=category,
            )

            yield query

    def cleanup(self) -> Iterator[DropTable]:
        """
        Cleanup the cohort definition combination by removing the temporary base table.
        """
        yield DropTable(self.base_table)

    @staticmethod
    def select_patients(category: CohortCategory) -> Select:
        """
        Select the patients in the given cohort category.
        """

        table = RecommendationResult.__table__

        return select(table.c.person_id).where(
            and_(
                table.c.recommendation_run_id == bindparam("run_id"),
                table.c.plan_id.is_(None),
                table.c.criterion_id.is_(None),
                table.c.cohort_category == category.name,
            )
        )

    def get_criterion(self, criterion_unique_name: str) -> Criterion:
        """
        Retrieve a criterion object by its unique name.
        """

        for cd in self._cohort_definitions:
            criterion = cd.get_criterion(criterion_unique_name)
            if criterion is not None:
                return criterion

        raise ValueError(f"Could not find criterion '{criterion_unique_name}'")

    def dict(self) -> dict:
        """
        Get the combination as a dictionary.
        """
        return {
            "id": self._id,
            "cohort_definitions": [c.dict() for c in self._cohort_definitions],
            "base_criterion": self._base_criterion.dict(),
            "recommendation_name": self._name,
            "recommendation_title": self._title,
            "recommendation_url": self._url,
            "recommendation_version": self._version,
            "recommendation_description": self._description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create a combination from a dictionary.
        """
        base_criterion = criterion_factory(**data["base_criterion"])
        assert isinstance(
            base_criterion, Criterion
        ), "Base criterion must be a Criterion"

        return cls(
            cohort_definitions=[
                CohortDefinition.from_dict(c) for c in data["cohort_definitions"]
            ],
            base_criterion=base_criterion,
            name=data["recommendation_name"],
            title=data["recommendation_title"],
            url=data["recommendation_url"],
            version=data["recommendation_version"],
            description=data["recommendation_description"],
            cohort_definition_id=data["id"] if "id" in data else None,
        )
