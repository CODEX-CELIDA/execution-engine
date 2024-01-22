from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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

        populate_by_name = True


class Settings(BaseSettings):  # type: ignore
    """Application settings."""

    timezone: str

    omop: OMOPSettings

    class Config:
        """
        Configuration for the settings.
        """

        env_file = "../.env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        env_prefix = "celida_ee_"


config = Settings()
