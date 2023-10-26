"""Some functions to update public resources in the kazu model pack."""
import argparse
import shutil
import sqlite3
import subprocess
from pathlib import Path

import requests

_ONTOLOGY_DIR_NAME = "ontologies"


def download_and_process_chembl(output_dir: Path, CHEMBL_VERSION: str) -> None:
    target_dir = output_dir.joinpath(_ONTOLOGY_DIR_NAME).absolute()
    chembl_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_{CHEMBL_VERSION}_sqlite.tar.gz".format(
        CHEMBL_VERSION=CHEMBL_VERSION
    )
    print("downloading chembl")
    subprocess.run(
        ["wget", chembl_url],
        cwd=target_dir,
    )
    print("extracting chembl DB")
    subprocess.run(["tar", "-xvzf", chembl_url.split("/")[-1]], cwd=target_dir)
    chembl_db_path = (
        target_dir.joinpath("chembl_{CHEMBL_VERSION}".format(CHEMBL_VERSION=CHEMBL_VERSION))
        .joinpath("chembl_{CHEMBL_VERSION}_sqlite".format(CHEMBL_VERSION=CHEMBL_VERSION))
        .joinpath("chembl_{CHEMBL_VERSION}.db".format(CHEMBL_VERSION=CHEMBL_VERSION))
    )

    conn = sqlite3.connect(chembl_db_path)
    LIST_TABLES = """SELECT
        name
    FROM
        sqlite_schema
    WHERE
        type ='table' AND
        name NOT LIKE 'sqlite_%';"""
    cur = conn.cursor()
    tables = cur.execute(LIST_TABLES)
    print("removing superfluous chembl tables to save space")
    for table_tup in tables.fetchall():
        table_name = table_tup[0]
        if table_name not in {
            "molecule_atc_classification",
            "molecule_dictionary",
            "molecule_hierarchy",
            "molecule_synonyms",
        }:
            print(f"dropping table {table_name}")
            drop_cur = conn.cursor()
            drop_cur.execute(f"DROP TABLE {table_name}")

            drop_cur.close()
    cur.close()
    print("running sqllite VACUUM")
    subprocess.run(
        ["sqlite3", chembl_db_path.name, "VACUUM;"],
        cwd=chembl_db_path.parent,
    )


def download_single_file_resources(output_dir: Path) -> None:
    ontologies_to_path = {
        "http://purl.obolibrary.org/obo/cl.owl": "cl.owl",
        "http://purl.obolibrary.org/obo/clo.owl": "clo.owl",
        "http://purl.obolibrary.org/obo/go.owl": "go.owl",
        "http://purl.obolibrary.org/obo/mondo.json": "mondo.json",
        "http://purl.obolibrary.org/obo/stato.owl": "stato.owl",
        "http://purl.obolibrary.org/obo/uberon.owl": "uberon.owl",
        'http://biomart.genenames.org/martservice/results?query=<!DOCTYPE Query><Query client="biomartclient" processor="TSV" limit="-1" header="1"><Dataset name="hgnc_family_mart" config="family_config"><Attribute name="family__family_id_103"/><Attribute name="family__name_103"/><Attribute name="family__alias__alias_101"/><Attribute name="family__root_symbol_103"/></Dataset></Query>': "gene_family.tsv",
        "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo": "cellosaurus.obo",
        "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp-full.json": "hp-full.json",
    }
    for url, local_path in ontologies_to_path.items():
        response = requests.get(url)
        if response.status_code > 300:
            print(f"request failed: {url}")
            print(response)
        else:
            print(f"downloading data from {url}")
            with output_dir.joinpath(_ONTOLOGY_DIR_NAME).joinpath(local_path).open(mode="w") as f:
                f.write(response.text)


def download_ftp_resources(output_dir: Path, OT_VERSION: str) -> None:
    path_to_ftp_directories = {
        "opentargets/": [
            "ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_VERSION}/output/etl/json/targets".format(
                OT_VERSION=OT_VERSION
            ),
            "ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_VERSION}/output/etl/json/diseases".format(
                OT_VERSION=OT_VERSION
            ),
            "ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_VERSION}/output/etl/json/molecule".format(
                OT_VERSION=OT_VERSION
            ),
        ]
    }
    for path_str, ftp_directories in path_to_ftp_directories.items():
        shutil.rmtree(path_str, ignore_errors=True)
        output_path = output_dir.joinpath(_ONTOLOGY_DIR_NAME).joinpath(path_str)
        output_path.mkdir(parents=True, exist_ok=True)
        for url in ftp_directories:

            subprocess.run(
                [
                    "wget",
                    "--recursive",
                    "--no-parent",
                    "--no-host-directories",
                    "--cut-dirs",
                    "8",
                    url,
                ],
                cwd=output_path,
            )


if __name__ == "__main__":
    description = """Download public ontologies for Kazu."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--open_targets_release_version",
        type=str,
        required=False,
        help="""Version of OpenTargets to use, e.g. '23.09'""",
    )
    parser.add_argument(
        "--chembl_version",
        type=str,
        required=False,
        help="""Version of Chembl to use, e.g. '33' """,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="""path to download resources to""",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    download_single_file_resources(output_dir)

    if args.open_targets_release_version:
        download_ftp_resources(output_dir, args.open_targets_release_version)
    else:
        print("skipping open targets")

    if args.chembl_version:
        download_and_process_chembl(output_dir, args.chembl_version)
    else:
        print("skipping chembl")
