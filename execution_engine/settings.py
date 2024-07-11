import datetime
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OMOPSettings(BaseModel):
    """
    OMOP database connection parameters.
    """

    host: str
    port: int
    user: str
    password: str
    database: str
    db_data_schema: str = Field(alias="data_schema", default="cds_cdm")
    db_result_schema: str = Field(alias="result_schema", default="celida")
    model_config = ConfigDict(populate_by_name=True)


class TimeInterval(BaseModel):
    """
    Time interval.
    """

    start: datetime.time
    end: datetime.time


class TimeIntervalSettings(BaseModel):
    """
    Time interval settings.
    """

    morning_shift: TimeInterval = TimeInterval(
        start=datetime.time(6, 0, 0), end=datetime.time(13, 59, 59)
    )
    afternoon_shift: TimeInterval = TimeInterval(
        start=datetime.time(14, 0, 0), end=datetime.time(21, 59, 59)
    )
    night_shift: TimeInterval = TimeInterval(
        start=datetime.time(22, 0, 0), end=datetime.time(5, 59, 59)
    )
    day: TimeInterval = TimeInterval(
        start=datetime.time(0, 0, 0), end=datetime.time(23, 59, 59)
    )


class Settings(BaseSettings):  # type: ignore
    """Application settings."""

    fhir_terminology_server_url: str = "http://tx.fhir.org/r4"
    fhir_base_url: str

    timezone: str = "Europe/Berlin"
    multiprocessing_use: bool = False
    episode_of_care_visit_detail: bool = False
    multiprocessing_pool_size: int = -1

    omop: OMOPSettings
    time_intervals: TimeIntervalSettings = TimeIntervalSettings()

    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="celida_ee_",
    )


_current_config = None


def get_config() -> Settings:
    """
    Returns the current configuration.
    """
    global _current_config

    if _current_config is None:
        _current_config = Settings()

    return _current_config


def update_config(**kwargs: Any) -> None:
    """
    Sets the current configuration with validation.
    """
    global _current_config

    if _current_config is None:
        _current_config = Settings()

    current_config_dict = _current_config.model_dump()

    updated_config_dict = {**current_config_dict, **kwargs}

    _current_config = Settings.model_validate(updated_config_dict)
