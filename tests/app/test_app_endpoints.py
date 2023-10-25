from datetime import datetime

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.app_state import AppState
from app.main import app
from app.schemas.comment import CommentCreate
from execution_engine.omop.db.cdm import Person
from execution_engine.omop.db.celida import Comment


@pytest.mark.recommendation
class TestAppEndpoints:
    @pytest.fixture(scope="class", autouse=True)
    def startup_event(self):
        AppState.initialize()

    @pytest.fixture
    def client(self, db_session):
        p = Person(
            person_id=1,
            gender_concept_id=0,
            year_of_birth=1990,
            race_concept_id=0,
            ethnicity_concept_id=0,
        )
        c = Comment(person_id=1, text="test comment", datetime=datetime.now())
        db_session.add_all([p, c])
        db_session.commit()

        with TestClient(app) as client:
            yield client

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "CODEX-CELIDA Execution Engine"}

    def test_get_comments(self, client):
        # Assuming there is a person with id 1 and he/she has comments
        response = client.get("/comments/1")
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
        # Further assertions can be done on the response content

    def test_get_comments_not_found(self, client):
        # Assuming there is no person with id 9999
        response = client.get("/comments/9999")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_comment(self, client):
        # Assuming there is a person with id 1
        comment_data = {
            "text": "test comment",
            "person_id": 1,
            "datetime": datetime.now(),
        }
        response = client.post("/comments/", data=CommentCreate(**comment_data).json())
        assert response.status_code == status.HTTP_200_OK
        # Further assertions can be done on the response content

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
