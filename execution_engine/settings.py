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
    db_schema: str = Field(alias="schema")

    class Config:
        """
        Pydantic config.
        """

        allow_population_by_field_name = True


class Settings(BaseSettings):
    """Application settings."""

    fhir_terminology_server_url: str = "http://tx.fhir.org/r4"
    fhir_base_url: str

    omop: OMOPSettings

    class Config:
        """
        Configuration for the settings.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


config = Settings()
