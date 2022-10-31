import os

from .client import WebAPIClient

webapi = WebAPIClient(os.environ["OMOP_WEBAPI_URL"])
