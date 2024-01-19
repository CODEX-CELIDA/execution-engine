from pydantic import BaseModel, BaseSettings, Field


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

    class Config:
        """
        Pydantic config.
        """

        allow_population_by_field_name = True


class Settings(BaseSettings):  # type: ignore
    """Application settings."""

    fhir_terminology_server_url: str = "http://tx.fhir.org/r4"
    fhir_base_url: str

    celida_ee_timezone: str
    celida_ee_multiprocessing_use: bool = False
    celida_ee_episode_of_care_visit_detail: bool = False
    celida_ee_multiprocessing_pool_size: int = 1

    omop: OMOPSettings

    class Config:
        """
        Configuration for the settings.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


_current_config = None


def get_config() -> Settings:
    """
    Returns the current configuration.
    """
    global _current_config

    if _current_config is None:
        _current_config = Settings()

    return _current_config


def set_config(config: Settings) -> None:
    """
    Sets the current configuration.
    """
    global _current_config

    _current_config = config
