import pytest

from execution_engine.omop.sqlclient import OMOPSQLClient
from execution_engine.omop.vocabulary import SNOMEDCT
from execution_engine.settings import config
from tests._fixtures.concept import (
    concept_covid19,
    concept_enoxparin,
    concept_enoxparin_ingredient,
    concept_heparin_ingredient,
    concepts_heparin_other,
)


class TestSQLClient:
    @pytest.fixture
    def sql_client(self, db_setup):
        return OMOPSQLClient(
            **config.omop.dict(by_alias=True),
            timezone=config.celida_ee_timezone,
            disable_foreign_key_checks=True
        )

    def test_get_concept_info(self, sql_client):
        with pytest.raises(AssertionError, match="Expected 1 Concept, got 0"):
            sql_client.get_concept_info(-1)

        c = sql_client.get_concept_info(concept_covid19.concept_id)
        assert c == concept_covid19

        c = sql_client.get_concept_info(concept_heparin_ingredient.concept_id)
        assert c == concept_heparin_ingredient

    def test_get_concept(self, sql_client):
        c = sql_client.get_concept(
            vocabulary=SNOMEDCT.omop_vocab_name, code="840539006"
        )
        assert c == concept_covid19

        with pytest.raises(AssertionError, match="Expected 1 Concept, got 0"):
            sql_client.get_concept(vocabulary=SNOMEDCT.omop_vocab_name, code="invalid")

        c = sql_client.get_concept(
            vocabulary=SNOMEDCT.omop_vocab_name, code="840539006", name="COVID-19"
        )
        assert c == concept_covid19

        with pytest.raises(AssertionError, match="Expected 1 Concept, got 0"):
            sql_client.get_concept(
                vocabulary=SNOMEDCT.omop_vocab_name, code="840539006", name="invalid"
            )

        c = sql_client.get_concept(
            vocabulary=SNOMEDCT.omop_vocab_name,
            code="840539006",
            name="COVID-19",
            standard=True,
        )
        assert c == concept_covid19

        c = sql_client.get_concept(
            vocabulary=SNOMEDCT.omop_vocab_name,
            code="116508003",
            name="Argatroban",
            standard=False,
        )
        assert c.concept_id == 4022726

        with pytest.raises(AssertionError, match="Expected 1 Concept, got 0"):
            sql_client.get_concept(
                vocabulary=SNOMEDCT.omop_vocab_name,
                code="116508003",
                name="Argatroban",
                standard=True,
            )

    def test_drug_vocabulary_to_ingredient(self, sql_client):
        for drug in concepts_heparin_other + [concept_heparin_ingredient]:
            ingr = sql_client.drug_vocabulary_to_ingredient_via_ancestor(
                drug.vocabulary_id, drug.concept_code
            )
            assert ingr == concept_heparin_ingredient

        for drug in [concept_enoxparin, concept_enoxparin_ingredient]:
            ingr = sql_client.drug_vocabulary_to_ingredient_via_ancestor(
                drug.vocabulary_id, drug.concept_code
            )
            assert ingr == concept_enoxparin_ingredient

    def test_drugs_by_ingredient(self, sql_client):
        # heparin
        df = sql_client.drugs_by_ingredient(concept_heparin_ingredient.concept_id)
        ingredient_concept = sql_client.get_drug_concept_info(
            concept_heparin_ingredient.concept_id
        )

        for drug in concepts_heparin_other + [concept_heparin_ingredient]:
            drug_concept = sql_client.get_drug_concept_info(drug.concept_id)
            if (
                ingredient_concept["amount_unit_concept_id"]
                == drug_concept["amount_unit_concept_id"]
            ):
                assert drug.concept_id in df.drug_concept_id.values
            else:
                assert drug.concept_id not in df.drug_concept_id.values

    def test_concept_related_to(self, sql_client):
        # some examples from the test database
        assert sql_client.concept_related_to(4022726, 1322207, "Maps to")
        assert sql_client.concept_related_to(4136433, 1308473, "Maps to")
        assert sql_client.concept_related_to(4156730, 1301065, "Maps to")
        assert sql_client.concept_related_to(4159321, 1301025, "Maps to")
        assert sql_client.concept_related_to(4159634, 1367571, "Maps to")

        # these examples are not in the test database
        return
        concept_id_heparin_drug = 1367697
        assert sql_client.concept_related_to(
            concept_id_heparin_drug, concept_heparin_ingredient.concept_id, "Ingredient"
        )

        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 19082103, "RxNorm has dose form"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 40160948, "Marketed form of"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 40820510, "Has supplier"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 43648125, "Marketed form of"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 40728101, "Marketed form of"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 995271, "Mapped from"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 995271, "Maps to"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 40719465, "Marketed form of"
        )
        assert sql_client.concept_related_to(
            concept_enoxparin.concept_id, 40756513, "Has brand name"
        )


""
