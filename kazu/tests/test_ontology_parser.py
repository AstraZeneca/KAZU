import dataclasses
import json
import tempfile
from itertools import chain
from pathlib import Path

import pytest

from kazu.data.data import CuratedTerm
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import CuratedTermDataset
from kazu.tests.utils import DummyParser


def test_injected_synonyms():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir).joinpath("injections.jsonl")
        missing_parser_name = "missing_parser_1"
        with open(path, "w") as f:

            first_injected_synonyms = CuratedTerm(
                term="hello I'm injected",
                action="keep",
                case_sensitive=False,
                entity_class="injection_test",
                curated_id_mappings={
                    "injection_test_parser_1": "first",
                    "injection_test_parser_2": "second",
                },
            )
            second_injected_synonyms = CuratedTerm(
                term="hello I'm also injected",
                action="keep",
                case_sensitive=False,
                entity_class="injection_test",
                curated_id_mappings={
                    "injection_test_parser_1": "third",
                    "injection_test_parser_2": "alpha",
                    missing_parser_name: "blah",
                },
            )
            null_injected_synonyms = CuratedTerm(
                term="I don't match anything",
                action="keep",
                case_sensitive=False,
                entity_class="injection_test",
                curated_id_mappings={missing_parser_name: "blah"},
            )
            f.write(json.dumps(dataclasses.asdict(first_injected_synonyms)) + "\n")
            f.write(json.dumps(dataclasses.asdict(second_injected_synonyms)) + "\n")
            f.write(json.dumps(dataclasses.asdict(null_injected_synonyms)) + "\n")

        parser = DummyParser(
            name="injection_test_parser_1",
            in_path="",
            entity_class="injection_test",
            additional_synonyms_dataset=CuratedTermDataset(str(path)),
        )
        parser2 = DummyParser(
            name="injection_test_parser_2",
            in_path="",
            entity_class="injection_test",
            additional_synonyms_dataset=CuratedTermDataset(str(path)),
        )
        parser.populate_databases()
        parser2.populate_databases()
        db = SynonymDatabase()

        for term in [first_injected_synonyms, second_injected_synonyms, null_injected_synonyms]:
            for parser_name, idx in term.curated_id_mappings.items():
                if parser_name != missing_parser_name:
                    term_norms = db.get_syns_for_id(name=parser_name, idx=idx)
                    term_strings = set(
                        chain.from_iterable(
                            db.get(name=parser_name, synonym=x).terms for x in term_norms
                        )
                    )
                    assert term.term in term_strings
                else:
                    with pytest.raises(KeyError):
                        db.get_syns_for_id(name=parser_name, idx=idx)
