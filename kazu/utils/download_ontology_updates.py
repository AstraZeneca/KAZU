"""Some functions to update public resources in the kazu model pack."""
import argparse
import shutil
import sqlite3
import subprocess
from pathlib import Path

import requests

_ONTOLOGY_DIR_NAME = "ontologies"

ONTOLOGY_NAMES_TO_PATH = {
    "CL": (
        "http://purl.obolibrary.org/obo/cl.owl",
        "cl.owl",
    ),
    "CLO": (
        "http://purl.obolibrary.org/obo/clo.owl",
        "clo.owl",
    ),
    "GO": (
        "http://purl.obolibrary.org/obo/go.owl",
        "go.owl",
    ),
    "MONDO": (
        "http://purl.obolibrary.org/obo/mondo.json",
        "mondo.json",
    ),
    "STATO": (
        "http://purl.obolibrary.org/obo/stato.owl",
        "stato.owl",
    ),
    "UBERON": (
        "http://purl.obolibrary.org/obo/uberon.owl",
        "uberon.owl",
    ),
    "HGNC_GENE_FAMILY": (
        "http://biomart.genenames.org/martservice/results?query="
        '<!DOCTYPE Query><Query client="biomartclient" processor="TSV" limit="-1" header="1">'
        '<Dataset name="hgnc_family_mart" config="family_config">'
        '<Attribute name="family__family_id_103"/><Attribute name="family__name_103"/>'
        '<Attribute name="family__alias__alias_101"/><Attribute name="family__root_symbol_103"/><'
        "/Dataset></Query>",
        "gene_family.tsv",
    ),
    "CELLOSAURUS": (
        "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo",
        "cellosaurus.obo",
    ),
    "HPO": (
        "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp-full.owl",
        "hp-full.owl",
    ),
}

DOWNLOAD_KEYS = list(ONTOLOGY_NAMES_TO_PATH.keys()) + ["OPENTARGETS", "CHEMBL"]


def download_and_process_chembl(output_dir: Path, chembl_version: str) -> None:
    target_dir = output_dir.joinpath(_ONTOLOGY_DIR_NAME).absolute()
    chembl_url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{chembl_version}/chembl_{chembl_version}_sqlite.tar.gz"

    print("downloading chembl")
    subprocess.run(
        ["wget", chembl_url],
        cwd=target_dir,
    )
    print("extracting chembl DB")
    subprocess.run(["tar", "-xvzf", chembl_url.split("/")[-1]], cwd=target_dir)
    chembl_db_path = (
        target_dir.joinpath(f"chembl_{chembl_version}")
        .joinpath(f"chembl_{chembl_version}_sqlite")
        .joinpath(f"chembl_{chembl_version}.db")
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
            conn.execute(f"DROP TABLE {table_name}")
    cur.close()
    print("running sqllite VACUUM")
    conn = sqlite3.connect(chembl_db_path, isolation_level=None)
    conn.execute("VACUUM")
    conn.close()


def download_single_file_resources(output_dir: Path, resources_to_download: list[str]) -> None:
    for resource_to_download in resources_to_download:
        maybe_url_path = ONTOLOGY_NAMES_TO_PATH.get(resource_to_download)
        if maybe_url_path is not None:
            url, local_path = maybe_url_path
            print(f"downloading data from {url}")
            response = requests.get(url, stream=True)
            if response.status_code > 300:
                print(f"request failed: {url}")
                print(response)
            else:
                with output_dir.joinpath(_ONTOLOGY_DIR_NAME).joinpath(local_path).open(
                    mode="wb"
                ) as f:
                    for chunk in response.iter_content(chunk_size=None):
                        f.write(chunk)


def download_ftp_resources(output_dir: Path, open_targets_version: str) -> None:
    path_to_ftp_directories = {
        "opentargets/": [
            f"ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{open_targets_version}/output/etl/json/targets"
            f"ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{open_targets_version}/output/etl/json/diseases"
            f"ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{open_targets_version}/output/etl/json/molecule"
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
    description = "Download public ontologies for Kazu."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--open_targets_release_version",
        type=str,
        required=False,
        help="Version of OpenTargets to use, e.g. '23.09'",
    )
    parser.add_argument(
        "--chembl_version",
        type=str,
        required=False,
        help="Version of Chembl to use, e.g. '33'",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to download resources to",
    )
    parser.add_argument(
        "--resources_to_download",
        nargs="+",
        required=False,
        default=DOWNLOAD_KEYS,
        help=f"resources to download. Choose from {DOWNLOAD_KEYS}. If blank, download all",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.joinpath(_ONTOLOGY_DIR_NAME).mkdir(parents=True, exist_ok=True)
    download_single_file_resources(output_dir, args.resources_to_download)

    if "OPENTARGETS" in args.resources_to_download:
        if args.open_targets_release_version:
            download_ftp_resources(output_dir, args.open_targets_release_version)
        else:
            print(
                "skipping open targets as no version was provided with --open_targets_release_version"
            )
    if "CHEMBL" in args.resources_to_download:
        if args.chembl_version:
            download_and_process_chembl(output_dir, args.chembl_version)
        else:
            print("skipping chembl as no version was provided with --chembl_version")
