"""This script does the following things:

1. Query Ensembl to get gene to protein ID maps
2. Query wikidata sparql to get a list of wikidata ids to Ensembl gene or Ensembl protein IDs
3. Query wikidata api to get a list of wikipedia page urls with the wikidata IDs from 2
4. Query Wikipedia API to get page content for each page from 3
5. Join wiki page content to Ensembl gene ids based on above relationships
"""

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from diskcache import Cache
from kazu.utils.caching import CacheProtocol
from pandas import Index
from tqdm import tqdm

try:
    import mwparserfromhell
except ImportError:
    raise ImportError("this script requires mwparserfromhell to be installed")

import pandas as pd
import requests

from requests.adapters import HTTPAdapter, Retry

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
BIOMART_URL = "http://www.ensembl.org/biomart/martservice"

WIKIDATA_SPARQL_ENSEMBL_GENE = r"""
SELECT DISTINCT ?item ?itemLabel ?Ensembl_gene_ID WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
  {
    SELECT DISTINCT ?item WHERE {
      ?item p:P594 ?statement0.
      ?statement0 ps:P594 _:anyValueP594.
    }
  }
  OPTIONAL { ?item wdt:P594 ?Ensembl_gene_ID. }
  FILTER STRSTARTS(?Ensembl_gene_ID, "ENSG")
}"""
WIKIDATA_SPARQL_ENSEMBL_PROTEIN = r"""
SELECT DISTINCT ?item ?itemLabel ?Ensembl_protein_ID WHERE {
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
  {
    SELECT DISTINCT ?item WHERE {
      ?item p:P705 ?statement0.
      ?statement0 ps:P705 _:anyValueP705.
    }
  }
  OPTIONAL { ?item wdt:P705 ?Ensembl_protein_ID. }
  FILTER STRSTARTS(?Ensembl_protein_ID, "ENSP")
}"""


BIOMART_GENE_TO_PROTEIN_QUERY = r"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "CSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
    <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "ensembl_peptide_id" />
    </Dataset>
</Query>"""

# wikidata allows up to 50 ids per request
WIKIPEDIA_API_CHUNK_SIZE = 50

WIKIPEDIA_URL_PREFIX = "https://en.wikipedia.org/wiki/"


def get_retry() -> Retry:
    return Retry(
        total=100, backoff_factor=0.2, connect=100, redirect=100, read=100, status=100, other=100
    )


cache: CacheProtocol = Cache(directory="gilda_contexts_cache")
wiki_etiquette_headers = {
    "Accept-Encoding": "gzip",
    "user_agent": "Kazu (https://github.com/AstraZeneca/KAZU)",
}


@dataclasses.dataclass
class WikipediaEnsemblMapping:
    ensembl_gene_id: str
    ensembl_protein_ids: set[str] = dataclasses.field(default_factory=set)
    wiki_gene_ids: set[str] = dataclasses.field(default_factory=set)
    wiki_protein_ids: set[str] = dataclasses.field(default_factory=set)
    wiki_gene_urls_to_text: dict[str, Optional[str]] = dataclasses.field(default_factory=dict)
    wiki_protein_urls_to_text: dict[str, Optional[str]] = dataclasses.field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(id(self))

    def get_context(self) -> Optional[str]:
        items = []
        for x in self.wiki_gene_urls_to_text.values():
            if x is not None:
                items.append(x)
        for x in self.wiki_protein_urls_to_text.values():
            if x is not None:
                items.append(x)
        if len(items) > 0:
            return "\n****\n".join(items)
        return None


@cache.memoize(ignore={1})
def get_sparql_df(query: str, proxies: dict[str, str]) -> pd.DataFrame:
    print(f"downloading wikidata ids for \n{query}")
    r = requests.get(
        WIKIDATA_SPARQL_URL,
        params={"query": query},
        headers={"Accept": "text/csv"},
        proxies=proxies,
    )
    return pd.read_csv(StringIO(r.content.decode("utf-8")))


@cache.memoize(ignore={0})
def get_biomart_gene_to_protein(proxies: dict[str, str]) -> pd.DataFrame:
    print("downloading Ensembl gene protein mappings")
    r = requests.get(BIOMART_URL, params={"query": BIOMART_GENE_TO_PROTEIN_QUERY}, proxies=proxies)
    df = pd.read_csv(StringIO(r.text))
    df.columns = Index(["gene_id", "protein_id"])
    return df


def divide_chunks(items: list[str]) -> Iterable[list[str]]:
    for i in range(0, len(items), WIKIPEDIA_API_CHUNK_SIZE):
        yield items[i : i + WIKIPEDIA_API_CHUNK_SIZE]


@cache.memoize(ignore={2})
def retry_wiki_with_maxlag(
    url: str, params: dict[str, str], proxies: dict[str, str]
) -> dict[str, Any]:
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=get_retry()))
    count = 1
    backoff = 2
    while True:
        response = s.get(url, params=params, proxies=proxies, headers=wiki_etiquette_headers)
        json_response: dict[str, Any] = response.json()
        error = json_response.get("error")
        if error is None:
            return json_response
        elif not error.get("code") == "maxlag":
            raise Exception(f"unknown API error: {json_response}")
        else:
            wait_for = int(response.headers.get("Retry-After", count * backoff))
            print(f"wiki api awaiting maxlag: backoff {wait_for}")
            time.sleep(wait_for)
            count += 1


@cache.memoize(ignore={0, 1, 2})
def get_wikipedia_url_from_wikidata_id(
    df_genes: pd.DataFrame, df_proteins: pd.DataFrame, proxies: dict[str, str]
) -> defaultdict[str, set[str]]:
    print("downloading wikipedia urls")
    wikidata_ids = set(df_genes["itemLabel"].tolist())
    wikidata_ids.update(df_proteins["itemLabel"].tolist())

    len_ids = len(wikidata_ids)
    len_chunks = len_ids // WIKIPEDIA_API_CHUNK_SIZE + bool(len_ids % WIKIPEDIA_API_CHUNK_SIZE)
    result = defaultdict(set)
    for chunk in tqdm(divide_chunks(sorted(wikidata_ids)), total=len_chunks):
        params = {
            "action": "wbgetentities",
            "props": "sitelinks/urls",
            "ids": "|".join(chunk),
            "format": "json",
            "maxlag": "3",
        }
        json_response = retry_wiki_with_maxlag(url=WIKIDATA_API_URL, params=params, proxies=proxies)
        entities = json_response.get("entities")
        if entities:
            for idx in wikidata_ids:
                entity = entities.get(idx)
                if entity:
                    sitelinks = entity.get("sitelinks")
                    if sitelinks:
                        # filter only the specified language
                        sitelink = sitelinks.get("enwiki")
                        if sitelink:
                            wiki_url = sitelink.get("url")
                            if wiki_url:
                                result[idx].add(unquote(wiki_url))
    return result


@cache.memoize(ignore={0, 1})
def get_wikipedia_contents_from_urls(urls: set[str], proxies: dict[str, str]) -> dict[str, str]:
    print("downloading wikipedia page contents")
    len_urls = len(urls)
    len_chunks = len_urls // WIKIPEDIA_API_CHUNK_SIZE + bool(len_urls % WIKIPEDIA_API_CHUNK_SIZE)
    # ignore these sections as they add noise
    exclude_sections = {"further reading", "references", "external links"}
    results = {}
    for chunk in tqdm(divide_chunks(sorted(urls)), total=len_chunks):

        params = {
            "action": "query",
            "prop": "revisions",
            "titles": "|".join(x.removeprefix(WIKIPEDIA_URL_PREFIX) for x in chunk),
            "rvprop": ["content"],
            "rvslots": ["main"],
            "format": "json",
            "maxlag": "3",
        }
        json_response = retry_wiki_with_maxlag(
            url=WIKIPEDIA_API_URL, params=params, proxies=proxies
        )
        query = json_response["query"]
        normalised = {x["to"]: x["from"] for x in query.get("normalized", [])}
        for page in query["pages"].values():
            title = page["title"]
            norm_title = normalised.get(title, title)
            for revision in page["revisions"]:
                main_slot = revision.get("slots", {}).get("main", {})
                wikitext = main_slot.get("*")
                if wikitext is not None:
                    parsed = mwparserfromhell.parse(wikitext)
                    sections = parsed.get_sections()
                    result = []
                    for wikisection in sections:
                        text = wikisection.strip_code()
                        text_lower = text.lower()
                        if not any(x in text_lower[:70] for x in exclude_sections):
                            result.append(text)
                    final_text = "\n***\n".join(result)
                    results[norm_title] = final_text
    return results


def create_wiki_mappings(
    gene_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    ensembl_gene_to_protein_mappings: pd.DataFrame,
    wikidata_id_to_wikipedia_urls: defaultdict[str, set[str]],
    wikipage_to_text: dict[str, str],
) -> set[WikipediaEnsemblMapping]:
    mappings_by_gene: dict[str, WikipediaEnsemblMapping] = dict()
    mappings_by_protein: dict[str, WikipediaEnsemblMapping] = dict()
    print("mapping ensembl genes to proteins")
    for i, row in tqdm(
        ensembl_gene_to_protein_mappings.iterrows(), total=ensembl_gene_to_protein_mappings.shape[0]
    ):
        ensembl_gene_id = row["gene_id"]
        mapping = mappings_by_gene.get(ensembl_gene_id)
        if mapping is None:
            mapping = WikipediaEnsemblMapping(ensembl_gene_id=ensembl_gene_id)
            mappings_by_gene[ensembl_gene_id] = mapping

        ensembl_protein_id = row["protein_id"]
        # check result is not NAN or another value we don't want
        if isinstance(ensembl_protein_id, str) and ensembl_protein_id.startswith("ENSP"):
            mapping.ensembl_protein_ids.add(ensembl_protein_id)
            mappings_by_protein[ensembl_protein_id] = mapping
    print("mapping wikipedia gene text to ensembl genes")
    for i, row in tqdm(gene_df.iterrows(), total=gene_df.shape[0]):
        ensembl_id = row["Ensembl_gene_ID"]
        wiki_id = row["item"].split("/")[-1]
        mapping = mappings_by_gene.get(ensembl_id)
        if mapping is not None:
            mapping.wiki_gene_ids.add(wiki_id)
            wikipedia_urls = wikidata_id_to_wikipedia_urls.get(wiki_id, set())
            for wikipedia_url in wikipedia_urls:
                page_name = wikipedia_url.removeprefix(WIKIPEDIA_URL_PREFIX)
                mapping.wiki_gene_urls_to_text[wikipedia_url] = wikipage_to_text.get(page_name)
    print("mapping wikipedia protein text to ensembl genes")
    for i, row in tqdm(protein_df.iterrows(), total=protein_df.shape[0]):
        ensembl_id = row["Ensembl_protein_ID"]
        wiki_id = row["item"].split("/")[-1]
        mapping = mappings_by_protein.get(ensembl_id)
        if mapping is not None:
            mapping.wiki_protein_ids.add(wiki_id)
            wikipedia_urls = wikidata_id_to_wikipedia_urls.get(wiki_id, set())
            for wikipedia_url in wikipedia_urls:
                page_name = wikipedia_url.removeprefix(WIKIPEDIA_URL_PREFIX)
                mapping.wiki_protein_urls_to_text[wikipedia_url] = wikipage_to_text.get(page_name)

    return set(mappings_by_gene.values())


def extract_open_targets(path: Path, proxies: dict[str, str]) -> None:
    ensembl_gene_to_protein_mappings = get_biomart_gene_to_protein(proxies)

    gene_df_humans = get_sparql_df(WIKIDATA_SPARQL_ENSEMBL_GENE, proxies)

    protein_df_humans = get_sparql_df(WIKIDATA_SPARQL_ENSEMBL_PROTEIN, proxies)

    wikidata_id_to_wikipedia_urls = get_wikipedia_url_from_wikidata_id(
        gene_df_humans, protein_df_humans, proxies
    )

    wiki_urls = set().union(wikidata_id_to_wikipedia_urls.values())

    page_name_to_text = get_wikipedia_contents_from_urls(wiki_urls, proxies)

    mappings = create_wiki_mappings(
        gene_df=gene_df_humans,
        protein_df=protein_df_humans,
        ensembl_gene_to_protein_mappings=ensembl_gene_to_protein_mappings,
        wikidata_id_to_wikipedia_urls=wikidata_id_to_wikipedia_urls,
        wikipage_to_text=page_name_to_text,
    )

    result_dict = {}
    for mapping in mappings:
        context = mapping.get_context()
        if context is not None:
            result_dict[mapping.ensembl_gene_id] = context

    if path.exists():
        with path.open(mode="r") as f:
            existing_json_dict = json.load(f)
    else:
        existing_json_dict = {}
    existing_json_dict["OPENTARGETS_TARGET"] = result_dict

    with path.open(mode="w") as f:
        json.dump(existing_json_dict, f, sort_keys=True, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Download contexts for gilda style disambiguation",
        description="Gilda TFIDF disambiguation requires contexts for every KB identifier, in order to disambiguate. This"
        " program queries external APIs to retrieve these",
    )
    parser.add_argument(
        "--parser_name",
        required=True,
        choices=["OPENTARGETS_TARGET"],
        help="parser to download contexts for",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="Path to write results to. If this path exists, the script will attempt to override only the contexts for the specified parser name.",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="clear a local cache to force re-download of all data",
    )
    parser.add_argument(
        "--http_proxy",
        required=False,
        help="http proxy address, if required",
    )
    parser.add_argument(
        "--https_proxy",
        required=False,
        help="https proxy address, if required",
    )

    args = parser.parse_args()
    proxies: dict[str, str] = {}
    if args.clear_cache:
        cache.clear()
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy
    if args.parser_name == "OPENTARGETS_TARGET":
        extract_open_targets(args.output_path, proxies)
