import pendulum
import pytest

from execution_engine.omop.concepts import Concept, CustomConcept
from execution_engine.omop.criterion.custom import (
    TidalVolumePerIdealBodyWeight as TVPIBW,
)
from execution_engine.util import ValueNumber
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.functions import create_measurement

# todo: add a test for mulitple heights --> only the latest one should be used


class TestTidalVolumePerIdealBodyWeight(TestCriterion):
    @pytest.fixture
    def concept(self):
        return CustomConcept(
            name="Tidal volume / ideal body weight (ARDSnet)",
            concept_code="tvpibw",
            domain_id="Measurement",
            vocabulary_id="CODEX-CELIDA",
        )

    @pytest.fixture
    def criterion_class(self):
        return TVPIBW

    def create_value(
        self, visit_occurrence, concept_id, datetime, value, unit_concept_id
    ):
        value_as_concept_id = value.concept_id if isinstance(value, Concept) else None
        value_as_number = value if isinstance(value, float | int) else None

        return create_measurement(
            vo=visit_occurrence,
            measurement_concept_id=concept_id,
            measurement_datetime=datetime,
            value_as_number=value_as_number,
            value_as_concept_id=value_as_concept_id,
            unit_concept_id=unit_concept_id,
        )

    @pytest.mark.parametrize(
        "times",
        [  # datetime for body height / tidal volume (in that order)
            {"height": "2023-03-04 06:00:00+01:00", "tv": "2023-03-04 07:00:00+01:00"},
            {"height": "2023-03-04 06:00:00+01:00", "tv": "2023-03-04 05:00:00+01:00"},
        ],
        ids=["height before tv", "tv before height"],
    )
    @pytest.mark.parametrize("gender", ["male", "female", "unknown"])
    @pytest.mark.parametrize(
        "values,expected",
        [  # TV/BW = 5.9
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 71
                    ),
                    "tv": 418.9,
                },
                ["2023-03-04"],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 70
                    ),
                    "tv": 413,
                },
                ["2023-03-04"],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 69
                    ),
                    "tv": 407.1,
                },
                ["2023-03-04"],
            ],
            # TV/BW = 6.0
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 71
                    ),
                    "tv": 426,
                },
                ["2023-03-04"],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 70
                    ),
                    "tv": 420,
                },
                ["2023-03-04"],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 69
                    ),
                    "tv": 141,
                },
                ["2023-03-04"],
            ],
            # TV/BW = 6.1
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 71
                    ),
                    "tv": 433.1,
                },
                [],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 70
                    ),
                    "tv": 427,
                },
                [],
            ],
            [
                {
                    "height": lambda gender: TVPIBW.height_for_predicted_body_weight_ardsnet(
                        gender, 69
                    ),
                    "tv": 420.9,
                },
                [],
            ],
        ],
        ids=["tv/bw=5.9_"] * 3 + ["tv/bw=6.0_"] * 3 + ["tv/bw=6.1_"] * 3,
    )  # time ranges used in the database entry
    @pytest.mark.parametrize(
        "exclude", [True, False], ids=["exclude=True", "exclude=False"]
    )  # exclude used in the criterion
    @pytest.mark.parametrize(
        "threshold", [6], ids=["threshold=6"]
    )  # threshold used in the criterion
    def test_single_day(
        self,
        concept,
        times,
        gender,
        values,
        expected,
        threshold,
        person_visit,
        db_session,
        observation_window,
        exclude,
        criterion_execute_func,
    ):
        from execution_engine.clients import omopdb

        p, vo = person_visit[0]

        # update person's gender
        p.gender_concept_id = {
            "male": concepts.GENDER_MALE,
            "female": concepts.GENDER_FEMALE,
            "unknown": concepts.UNKNOWN,
        }[gender]
        db_session.add(p)

        v_height = self.create_value(
            visit_occurrence=vo,
            concept_id=concepts.BODY_HEIGHT,
            datetime=pendulum.parse(times["height"]),
            value=values["height"](gender),
            unit_concept_id=concepts.UNIT_CM,
        )
        v_tv = self.create_value(
            visit_occurrence=vo,
            concept_id=concepts.TIDAL_VOLUME,
            datetime=pendulum.parse(times["tv"]),
            value=values["tv"],
            unit_concept_id=concepts.UNIT_ML,
        )
        db_session.add_all([v_height, v_tv])
        db_session.commit()

        value = ValueNumber(
            value_max=threshold, unit=omopdb.get_concept_info(concepts.UNIT_ML_PER_KG)
        )

        # run criterion against db
        df = criterion_execute_func(
            concept=concept, value=value, exclude=exclude
        ).query(f"{p.person_id} == person_id")

        # exclusion is now performed only when combining the criteria into population/intervention
        # if exclude:
        #    expected = self.invert_date_points(
        #        time_range=observation_window,
        #        subtract=expected,
        #    )

        assert set(df["valid_date"].dt.date) == self.date_points(expected)

    def test_predicted_body_weight_ardsnet(self):
        # Test the function for a few basic cases to make sure it's working
        assert TVPIBW.predicted_body_weight_ardsnet("male", 122) == pytest.approx(
            22.4, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("male", 160) == pytest.approx(
            56.9, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("male", 190) == pytest.approx(
            84.5, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("male", 213) == pytest.approx(
            105.2, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("female", 122) == pytest.approx(
            17.9, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("female", 160) == pytest.approx(
            52.4, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("female", 190) == pytest.approx(
            80.0, 0.1
        )
        assert TVPIBW.predicted_body_weight_ardsnet("female", 213) == pytest.approx(
            100.7, 0.1
        )

        # Test the function for some edge cases
        assert TVPIBW.predicted_body_weight_ardsnet("female", 0) == pytest.approx(
            -93.184, 0.01
        )
        assert TVPIBW.predicted_body_weight_ardsnet("male", 0) == pytest.approx(
            -88.684, 0.01
        )

        # Test the function for unrecognized genders
        with pytest.raises(ValueError):
            TVPIBW.predicted_body_weight_ardsnet("other", 180)

        # Test the function for some invalid inputs
        with pytest.raises(ValueError):
            TVPIBW.predicted_body_weight_ardsnet(1, 180)
            TVPIBW.predicted_body_weight_ardsnet("male", "180")

    def test_height_for_predicted_body_weight_ardsnet(self):
        # Test the function for a few basic cases to make sure it's working
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "male", 22.4
        ) == pytest.approx(122, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "male", 56.9
        ) == pytest.approx(160, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "male", 84.5
        ) == pytest.approx(190, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "male", 105.2
        ) == pytest.approx(213, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", 17.9
        ) == pytest.approx(122, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", 52.4
        ) == pytest.approx(160, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", 80.0
        ) == pytest.approx(190, 0.1)
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", 100.7
        ) == pytest.approx(213, 0.1)

        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "male",
                TVPIBW.predicted_body_weight_ardsnet("male", 122),
            )
            == 122
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "male",
                TVPIBW.predicted_body_weight_ardsnet("male", 160),
            )
            == 160
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "male",
                TVPIBW.predicted_body_weight_ardsnet("male", 190),
            )
            == 190
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "male",
                TVPIBW.predicted_body_weight_ardsnet("male", 213),
            )
            == 213
        )

        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "female",
                TVPIBW.predicted_body_weight_ardsnet("female", 122),
            )
            == 122
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "female",
                TVPIBW.predicted_body_weight_ardsnet("female", 160),
            )
            == 160
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "female",
                TVPIBW.predicted_body_weight_ardsnet("female", 190),
            )
            == 190
        )
        assert (
            TVPIBW.height_for_predicted_body_weight_ardsnet(
                "female",
                TVPIBW.predicted_body_weight_ardsnet("female", 213),
            )
            == 213
        )

        # Test the function for some edge cases
        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", -93.184 == pytest.approx(0, 0.01)
        )

        assert TVPIBW.height_for_predicted_body_weight_ardsnet(
            "female", -88.684 == pytest.approx(0, 0.01)
        )

        # Test the function for unrecognized genders
        with pytest.raises(ValueError):
            TVPIBW.height_for_predicted_body_weight_ardsnet("other", 76.42)

        # Test the function for some invalid inputs
        with pytest.raises(ValueError):
            TVPIBW.height_for_predicted_body_weight_ardsnet(1, 76.42)
            TVPIBW.height_for_predicted_body_weight_ardsnet("male", "76.42")
