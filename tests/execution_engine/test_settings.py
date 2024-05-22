import os

import pytest

from execution_engine import settings
from execution_engine.settings import Settings, get_config, update_config


class TestSettings:
    @pytest.fixture(autouse=True)
    def clean_settings(self):
        os.environ.pop("CELIDA_EE_MULTIPROCESSING_USE", None)
        settings._current_config = None

    def test_get_config(self):
        config = get_config()
        assert isinstance(
            config, Settings
        ), "get_config should return an instance of Settings"

    @pytest.mark.parametrize("initial", ["1", "0"])
    def test_env_config(self, initial):
        os.environ["CELIDA_EE_MULTIPROCESSING_USE"] = initial

        initial_config = get_config()

        assert initial_config.multiprocessing_use == bool(int(initial))

    def test_update_config(self):
        os.environ["CELIDA_EE_MULTIPROCESSING_USE"] = "0"

        initial_config = get_config()
        initial_dict = initial_config.model_dump()

        assert not initial_config.multiprocessing_use

        update_config(multiprocessing_use=True)
        updated_config = get_config()
        updated_dict = updated_config.model_dump()
        assert updated_dict[
            "multiprocessing_use"
        ], "update_config should update the configuration"
        assert (
            initial_dict != updated_dict
        ), "update_config should change the configuration"
