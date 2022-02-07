from pathlib import Path

import typer
import logging


from modelling.ontology_preprocessing.base import (
    UberonParser,
    MondoParser,
    CLOOntologyParser,
    ChemblOntologyParser,
    EnsemblOntologyParser,
    CellosaurusOntologyParser,
)


def main(raw_dir: str, processed_dir: str):
    ontologies_to_parquet(raw_dir, processed_dir)


def ontologies_to_parquet(raw_dir, processed_dir):
    Path(processed_dir).mkdir(exist_ok=True)
    _process_uberon(raw_dir, processed_dir)
    _process_mondo(raw_dir, processed_dir)
    _process_clo(raw_dir, processed_dir)
    _process_cellosaurus(raw_dir, processed_dir)
    _process_chembl(raw_dir, processed_dir)
    _process_ensembl(raw_dir, processed_dir)


def _process_uberon(raw_dir, processed_dir):
    logging.info("Fetching Uberon data")
    parser = UberonParser(in_path=f"{raw_dir}/uberon.owl")
    logging.info("Creating & writing synonyms for Uberon")
    parser.write_synonym_table(out_path=processed_dir)


def _process_mondo(raw_dir, processed_dir):
    logging.info("Fetching Mondo data")
    parser = MondoParser(in_path=f"{raw_dir}/mondo.json")
    logging.info("Creating & writing synonyms for Mondo")
    parser.write_synonym_table(out_path=processed_dir)


def _process_ensembl(raw_dir, processed_dir):
    logging.info("Fetching ENSEMBL data")
    parser = EnsemblOntologyParser(in_path=f"{raw_dir}/hgnc.json")
    logging.info("Creating & writing synonyms for ENSEMBL")
    parser.write_synonym_table(out_path=processed_dir)


def _process_chembl(raw_dir, processed_dir):
    logging.info("Fetching CHEMBL data")
    parser = ChemblOntologyParser(in_path=f"{raw_dir}/chembl_29_sqlite/chembl_29.db")
    logging.info("Creating & writing synonyms for CHEMBL")
    parser.write_synonym_table(out_path=processed_dir)


def _process_clo(raw_dir, processed_dir):
    logging.info("Fetching CLO data")
    parser = CLOOntologyParser(in_path=f"{raw_dir}/clo.owl")
    logging.info("Creating & writing synonyms for CLO")
    parser.write_synonym_table(out_path=processed_dir)


def _process_cellosaurus(raw_dir, processed_dir):
    logging.info("Fetching CELLOSAURUS data")
    parser = CellosaurusOntologyParser(in_path=f"{raw_dir}/cellosaurus.obo")
    logging.info("Creating & writing synonyms for CELLOSAURUS")
    parser.write_synonym_table(out_path=processed_dir)


if __name__ == "__main__":
    typer.run(main)
