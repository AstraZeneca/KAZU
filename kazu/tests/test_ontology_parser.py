import dataclasses
import json

import pytest

from kazu.data.data import CuratedTerm
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import CuratedTermDataset
from kazu.tests.utils import DummyParser


def test_injected_synonyms(tmp_path):
    path = tmp_path.joinpath("injections.jsonl")
    missing_parser_name = "missing_parser_1"

    injected_synonym_curated_terms = (
        CuratedTerm(
            term="hello I'm injected",
            action="keep",
            case_sensitive=False,
            entity_class="injection_test",
            curated_id_mappings={
                "injection_test_parser_1": "first",
                "injection_test_parser_2": "second",
            },
        ),
        CuratedTerm(
            term="hello I'm also injected",
            action="keep",
            case_sensitive=False,
            entity_class="injection_test",
            curated_id_mappings={
                "injection_test_parser_1": "third",
                "injection_test_parser_2": "alpha",
                missing_parser_name: "blah",
            },
        ),
        # won't match any parser
        CuratedTerm(
            term="I don't match anything",
            action="keep",
            case_sensitive=False,
            entity_class="injection_test",
            curated_id_mappings={missing_parser_name: "blah"},
        ),
    )

    with open(path, "w") as f:
        f.writelines(
            json.dumps(dataclasses.asdict(curated_term)) + "\n"
            for curated_term in injected_synonym_curated_terms
        )

    parser = DummyParser(
        name="injection_test_parser_1",
        in_path="",
        entity_class="injection_test",
        additional_synonyms_dataset=CuratedTermDataset(path),
    )
    parser2 = DummyParser(
        name="injection_test_parser_2",
        in_path="",
        entity_class="injection_test",
        additional_synonyms_dataset=CuratedTermDataset(path),
    )
    parser.populate_databases()
    parser2.populate_databases()
    db = SynonymDatabase()

    for term in injected_synonym_curated_terms:
        for parser_name, idx in term.curated_id_mappings.items():
            if parser_name != missing_parser_name:
                term_norms = db.get_syns_for_id(name=parser_name, idx=idx)
                term_strings = set(
                    term for x in term_norms for term in db.get(name=parser_name, synonym=x).terms
                )
                assert term.term in term_strings
            else:
                with pytest.raises(KeyError):
                    db.get_syns_for_id(name=parser_name, idx=idx)

