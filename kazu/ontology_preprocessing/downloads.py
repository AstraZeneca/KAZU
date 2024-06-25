"""Some functions to update public resources in the kazu model pack."""
import abc
import datetime
import functools
import logging
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional, cast

import rdflib
import requests
from rdflib.query import ResultRow

logger = logging.getLogger(__name__)


@functools.cache
def _cached_request(url: str, local_path: Path) -> None:
    logger.info(f"downloading data from {url}")
    response = requests.get(url, stream=True)
    if response.status_code > 300:
        logger.info("request failed: %s", url)
        logger.info(response)
    else:
        with local_path.open(mode="wb") as f:
            for chunk in response.iter_content(chunk_size=None):
                f.write(chunk)


@functools.cache
def _cached_wget(full_url: str, cwd: Path) -> None:
    subprocess.run(
        [
            "wget",
            "--recursive",
            "--no-parent",
            "--no-host-directories",
            "--cut-dirs",
            "8",
            full_url,
        ],
        cwd=cwd,
    )


class OntologyDownloader(abc.ABC):
    @abc.abstractmethod
    def download(self, local_path: Path, skip_download: bool = False) -> Path:
        """Download the ontology to the local path.

        :param local_path: the path to download the ontology to
        :param skip_download: whether to skip the download
        :return: the path to the downloaded ontology
        """
        pass

    def delete_previous(self, local_path: Path) -> None:
        """Delete the previous version of the ontology.

        :param local_path: the path to the ontology to delete
        """
        if local_path.exists():
            os.remove(local_path)

    @abc.abstractmethod
    def version(self, local_path: Optional[Path] = None) -> str:
        """Get the version of the ontology.

        Note that this method should be idempotent, i.e. it should not change the state
        of the ontology. Also, it may be able to determine the version of the ontology
        without querying it directly (e.g. by looking at the file name, or if it is
        known a priori). If this is not the case, you can implement a method here to do
        something more sophisticated, such as querying the ontology directly via sparql.

        :param local_path: the path to the ontology
        :return: the version of the ontology
        """
        pass


class SimpleOntologyDownloader(OntologyDownloader):
    def __init__(self, url: str):
        self.url = url

    def download(self, local_path: Path, skip_download: bool = False) -> Path:
        if not skip_download:
            _cached_request(self.url, local_path)
        return local_path

    def version(self, local_path: Optional[Path] = None) -> str:
        return datetime.datetime.now().isoformat()


class OBOOntologyDownloader(SimpleOntologyDownloader):
    def version(self, local_path: Optional[Path] = None) -> str:
        assert local_path is not None
        matcher = re.compile(r"data-version: (.*)$")
        for line in local_path.open(mode="r"):
            match = matcher.match(line)
            if match:
                return match.group(1)
        logger.warning("could not determine version for %s", local_path)
        return super().version(local_path)


class OwlOntologyDownloader(SimpleOntologyDownloader):
    """Used when version info is contained within the ontology."""

    _VERSION_INFO_QUERY = """SELECT ?aname
                    WHERE {
                        ?a rdf:type owl:Ontology .
                        ?a owl:versionInfo ?aname .
                    }"""

    _VERSION_IRI_QUERY = """SELECT ?aname
                    WHERE {
                        ?a rdf:type owl:Ontology .
                        ?a owl:versionIRI ?aname .
                    }"""

    def version(self, local_path: Optional[Path] = None) -> str:
        """Queries the Ontology for owl:versionInfo. Failing that, queries for
        owl:versionIRI. If it can't find it, it falls back to the superclass
        implementation.

        :param local_path:
        :return:
        """
        graph = rdflib.Graph().parse(local_path)
        v_info_result: list[ResultRow] = cast(
            list[ResultRow], list(graph.query(self._VERSION_INFO_QUERY))
        )
        if len(v_info_result) == 1:
            return str(v_info_result[0][0])
        else:
            v_iri_result: list[ResultRow] = cast(
                list[ResultRow], list(graph.query(self._VERSION_IRI_QUERY))
            )
            if len(v_iri_result) == 1:
                return str(v_iri_result[0][0])
            else:
                logger.warning("could not determine versionInfo for %s", local_path)
            return super().version()


class ChemblOntologyDownloader(OntologyDownloader):

    CHEMBL_TEMPLATE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_%s/chembl_%s_sqlite.tar.gz"

    def __init__(self, chembl_version: str):
        self.chembl_version = chembl_version

    def download(self, local_path: Path, skip_download: bool = False) -> Path:
        chembl_url = self.CHEMBL_TEMPLATE % (self.chembl_version, self.chembl_version)
        # since cheml comes as a zip, we use the parent path to extract it
        parent_path = local_path.parent
        chembl_zip_path = parent_path.joinpath(chembl_url.split("/")[-1])
        chembl_unzip_path = parent_path.joinpath(f"chembl_{self.chembl_version}")
        chembl_db_path = chembl_unzip_path.joinpath(
            f"chembl_{self.chembl_version}_sqlite"
        ).joinpath(f"chembl_{self.chembl_version}.db")
        new_path = parent_path.joinpath(chembl_db_path.name)
        if not skip_download:
            logger.info("downloading chembl %s", self.chembl_version)
            subprocess.run(
                ["wget", chembl_url],
                cwd=parent_path,
            )
            logger.info("extracting chembl DB")
            subprocess.run(["tar", "-xvzf", str(chembl_zip_path.absolute())], cwd=parent_path)
            os.remove(chembl_zip_path)
            # move it to the parent
            shutil.move(chembl_db_path, new_path)

            conn = sqlite3.connect(new_path)
            LIST_TABLES = """SELECT
                name
            FROM
                sqlite_schema
            WHERE
                type ='table' AND
                name NOT LIKE 'sqlite_%';"""
            cur = conn.cursor()
            tables = cur.execute(LIST_TABLES)
            logger.info("removing superfluous chembl tables to save space")
            for table_tup in tables.fetchall():
                table_name = table_tup[0]
                if table_name not in {
                    "molecule_atc_classification",
                    "molecule_dictionary",
                    "molecule_hierarchy",
                    "molecule_synonyms",
                }:
                    logger.info("dropping table %s", table_name)
                    conn.execute(f"DROP TABLE {table_name}")
            cur.close()
            logger.info("running sqllite VACUUM")
            conn = sqlite3.connect(chembl_db_path, isolation_level=None)
            conn.execute("VACUUM")
            conn.close()
        return new_path

    def version(self, local_path: Optional[Path] = None) -> str:
        return self.chembl_version

    def delete_previous(self, local_path: Path) -> None:
        for file in local_path.parent.glob("chembl_*.db"):
            if self.chembl_version not in file.name:
                os.remove(file)


class OpenTargetsOntologyDownloader(OntologyDownloader):
    OT_PREFIX = "ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/"

    def __init__(self, open_targets_version: str, open_targets_dataset_name: str):
        self.open_targets_dataset_name = open_targets_dataset_name
        self.open_targets_version = open_targets_version

    def download(self, local_path: Path, skip_download: bool = False) -> Path:
        if not skip_download:
            full_url = f"{self.OT_PREFIX}{self.open_targets_version}/output/etl/json/{self.open_targets_dataset_name}"
            _cached_wget(full_url, local_path.parent)
        return local_path

    def version(self, local_path: Optional[Path] = None) -> str:
        return self.open_targets_version

    def delete_previous(self, local_path: Path) -> None:
        """We could use rmtree here but it's safer to just remove the files we know we
        downloaded.

        :param local_path:
        :return:
        """
        if local_path.exists():
            for file in local_path.iterdir():
                if file.name.startswith("part-") or file.name.startswith("_SUCCESS"):
                    os.remove(file)
            os.removedirs(local_path)
