from typing import Callable

import spacy


BLACKLIST_EXACT = {
    "CHEMBL1201112": ["MAY"],
    "CLO_0054406": ["positive"],
    "CVCL_E025": ["Cancer"],
    "ENSG00000140254": ["mol"],
    "MONDO_0010518": ["WAS"],
    "MONDO_0000001": ["*"],
    "MONDO_0002254": ["*"],
    "MONDO_0009994": ["arms"],
    "MONDO_0021137": ["*"],
    "UBERON_0000105": ["stage"],
    "UBERON_0004529": ["*"],
    "UBERON_0006611": ["test"],
    "UBERON_0007023": ["*"],
    "*": ["was", "for"],
}

BLACKLIST_LOWER = {
    "CHEMBL2272076": ["impact"],
    "MONDO_0012268": ["aids"],
    "*": ["all", "was"],
}


@spacy.registry.misc("arizona.entry_filter_blacklist.v1")
def create_filter() -> Callable:
    return is_valid_ontology_entry


def is_valid_ontology_entry(row, lowercase):
    syn = row["syn"]
    iri = row["iri"]
    if len(syn) < 3:
        return False
    # we avoid case-invariant matching for some types of ontologies
    if lowercase and (iri.startswith("ENS") or iri.startswith("CVCL") or iri.startswith("CLO")):
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
    except:
        return False
