import pytest

from execution_engine.omop.criterion.custom import TidalVolumePerIdealBodyWeight


class TestTidalVolumePerIdealBodyWeight:
    def test_predicted_body_weight_ardsnet(self):
        # Test the function for a few basic cases to make sure it's working
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "male", 122
        ) == pytest.approx(22.4, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "male", 160
        ) == pytest.approx(56.9, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "male", 190
        ) == pytest.approx(84.5, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "male", 213
        ) == pytest.approx(105.2, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "female", 122
        ) == pytest.approx(17.9, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "female", 160
        ) == pytest.approx(52.4, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "female", 190
        ) == pytest.approx(80.0, 0.1)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "female", 213
        ) == pytest.approx(100.7, 0.1)

        # Test the function for some edge cases
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "female", 0
        ) == pytest.approx(-93.184, 0.01)
        assert TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
            "male", 0
        ) == pytest.approx(-88.684, 0.01)

        # Test the function for unrecognized genders
        with pytest.raises(ValueError):
            TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet("other", 180)

        # Test the function for some invalid inputs
        with pytest.raises(ValueError):
            TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(1, 180)
            TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet("male", "180")

    def test_height_for_predicted_body_weight_ardsnet(self):
        # Test the function for a few basic cases to make sure it's working
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "male", 22.4
        ) == pytest.approx(122, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "male", 56.9
        ) == pytest.approx(160, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "male", 84.5
        ) == pytest.approx(190, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "male", 105.2
        ) == pytest.approx(213, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", 17.9
        ) == pytest.approx(122, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", 52.4
        ) == pytest.approx(160, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", 80.0
        ) == pytest.approx(190, 0.1)
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", 100.7
        ) == pytest.approx(213, 0.1)

        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "male",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "male", 122
                ),
            )
            == 122
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "male",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "male", 160
                ),
            )
            == 160
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "male",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "male", 190
                ),
            )
            == 190
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "male",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "male", 213
                ),
            )
            == 213
        )

        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "female",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "female", 122
                ),
            )
            == 122
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "female",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "female", 160
                ),
            )
            == 160
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "female",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "female", 190
                ),
            )
            == 190
        )
        assert (
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "female",
                TidalVolumePerIdealBodyWeight.predicted_body_weight_ardsnet(
                    "female", 213
                ),
            )
            == 213
        )

        # Test the function for some edge cases
        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", -93.184 == pytest.approx(0, 0.01)
        )

        assert TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
            "female", -88.684 == pytest.approx(0, 0.01)
        )

        # Test the function for unrecognized genders
        with pytest.raises(ValueError):
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "other", 76.42
            )

        # Test the function for some invalid inputs
        with pytest.raises(ValueError):
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                1, 76.42
            )
            TidalVolumePerIdealBodyWeight.height_for_predicted_body_weight_ardsnet(
                "male", "76.42"
            )
