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
    data_schema: str = Field(alias="data_schema", default="cds_cdm")
    result_schema: str = Field(alias="result_schema", default="celida")
    model_config = ConfigDict(populate_by_name=True)


class Settings(BaseSettings):  # type: ignore
    """Application settings."""

    timezone: str

    omop: OMOPSettings
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="celida_ee_",
    )


config = Settings()
