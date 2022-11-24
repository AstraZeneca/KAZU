import json
import tempfile
from itertools import chain
from pathlib import Path

from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import IDX, SYN
from kazu.tests.utils import DummyParser


def test_injected_synonyms():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir).joinpath("injections.jsonl")
        with open(path, "w") as f:
            first_injected_synonyms = {IDX: "first", SYN: ["hello I'm injected", "so am I"]}
            second_injected_synonyms = {IDX: "second", SYN: ["I'm injected for second", "Also me"]}
            f.write(json.dumps(first_injected_synonyms) + "\n")
            f.write(json.dumps(second_injected_synonyms) + "\n")

        parser = DummyParser(
            name="injection_test",
            in_path="",
            entity_class="injection_test",
            additional_synonyms=str(path),
        )
        parser.populate_databases()
        db = SynonymDatabase()
        first_synonym_terms_norm = db.get_syns_for_id(name="injection_test", idx="first")
        first_all_terms = set(
            chain.from_iterable(
                db.get(name="injection_test", synonym=x).terms for x in first_synonym_terms_norm
            )
        )
        assert all(x in first_all_terms for x in first_injected_synonyms[SYN])
