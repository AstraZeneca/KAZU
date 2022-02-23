from typing import Callable

import spacy

from kazu.modelling.ontology_preprocessing.base import IDX, SYN


BLACKLIST_EXACT = {
    "CHEMBL1201112": ["MAY"],
    "http://purl.obolibrary.org/obo/CLO_0054406": ["positive"],
    "CVCL_E025": ["Cancer"],
    "ENSG00000140254": ["mol"],
    "http://purl.obolibrary.org/obo/MONDO_0010518": ["WAS"],
    "http://purl.obolibrary.org/obo/MONDO_0000001": ["*"],
    "http://purl.obolibrary.org/obo/MONDO_0002254": ["*"],
    "http://purl.obolibrary.org/obo/MONDO_0009994": ["arms"],
    "http://purl.obolibrary.org/obo/MONDO_0021137": ["*"],
    "http://purl.obolibrary.org/obo/UBERON_0000105": ["stage"],
    "http://purl.obolibrary.org/obo/UBERON_0004529": ["*"],
    "http://purl.obolibrary.org/obo/UBERON_0006611": ["test"],
    "http://purl.obolibrary.org/obo/UBERON_0007023": ["*"],
    "http://purl.obolibrary.org/obo/HP_0000001": ["All"],
    # a fair few entities have synonyms 'X disease' or 'X syndrome' where
    # x happens to also be a stopword. Getting rid of these here - we
    # may be able to use StopWordRemover more intelligently instead
    "*": ["was", "for", "disease", "Disease", "syndrome", "Syndrome"],
}

BLACKLIST_LOWER = {
    "CHEMBL2272076": ["impact"],
    "http://purl.obolibrary.org/obo/MONDO_0012268": ["aids"],
    # as above in BLACKLIST_EXACT
    "*": ["all", "was", "disease", "syndrome"],
}


@spacy.registry.misc("arizona.entry_filter_blacklist.v1")
def create_filter() -> Callable:
    return is_valid_ontology_entry


def is_valid_ontology_entry(row, lowercase):
    syn = row[SYN]
    iri = row[IDX]
    if len(syn) < 3:
        return False
    # we avoid case-invariant matching for some types of ontologies
    if lowercase and (
        iri.startswith("ENS")
        or iri.startswith("CVCL")
        or iri.startswith("http://purl.obolibrary.org/obo/CLO_")
    ):
        return False
    if _is_number(syn):
        return False
    if _hits_blacklist(BLACKLIST_EXACT, syn, iri):
        return False
    if lowercase:
        if _hits_blacklist(BLACKLIST_LOWER, syn.lower(), iri):
            return False
    return True


def _hits_blacklist(blacklist, syn, iri):
    if syn in blacklist.get(iri, []):
        return True
    if "*" in blacklist.get(iri, []):
        return True
    if syn in blacklist["*"]:
        return True
    return False


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
