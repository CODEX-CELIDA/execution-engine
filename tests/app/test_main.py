import pytest
from fastapi.testclient import TestClient

from app.app_state import AppState
from app.main import app


@pytest.mark.recommendation
class TestAppEndpoints:
    @pytest.fixture(scope="class", autouse=True)
    def startup_event(self):
        AppState.initialize()

    @pytest.fixture
    def client(self, db_session):
        with TestClient(app) as client:
            yield client

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "CODEX-CELIDA Execution Engine"}

    def test_recommendation_list(self, client):
        response = client.get("/recommendation/list")
        assert response.status_code == 200
        json = response.json()
        assert all(
            [
                set(rec.keys())
                == {
                    "recommendation_url",
                    "recommendation_name",
                    "recommendation_title",
                    "recommendation_description",
                }
                for rec in json
            ]
        )

    def test_recommendation_criteria(self, client):
        response = client.get("/recommendation/list")
        recommendations = response.json()
        recommendation_url = recommendations[0]["recommendation_url"]

        response = client.get(
            f"/recommendation/criteria?recommendation_url={recommendation_url}"
        )
        assert response.status_code == 200
        assert "criterion" in response.json()

    def test_patient_list(self, client):
        # TODO: Implement (need to fill database with actual data for this to work)
        pass

    def test_patient_data(self, client):
        # TODO: Implement (need to fill database with actual data for this to work)
        pass
