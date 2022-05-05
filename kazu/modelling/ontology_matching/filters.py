import re
from typing import Tuple

from kazu.data.data import SynonymData

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
    "*": ["all", "was", "disease", "syndrome", "cat", "mat", "has", "may", "can", "same", "read"],
}
# a set of gene names that are often used to mean 'actual genes' but also other things (such as page numbers, in the
# case of p27, p53 etc). We match on them, and disambiguate later
PROBLEMATIC_GENE_NAMES = {r"p[0-9][0-9]"}
PROBLEMATIC_GENE_NAMES_RE = [re.compile(x) for x in PROBLEMATIC_GENE_NAMES]


def is_valid_ontology_entry(syn: str, idx_str: str) -> Tuple[bool, bool]:
    """Returns a tuple of two bools:
    whether to add an case-sensitive match and whether to add a case-invariant match.
    We don't need a case-sensitive match rule if there's a case-invariant one,
    so returning (True, True) is an invalid input
    """
    if len(syn) < 3:
        return False, False
    if _is_number(syn):
        return False, False
    if _hits_blacklist(BLACKLIST_EXACT, syn, idx_str):
        return False, False

    if _hits_blacklist(BLACKLIST_LOWER, syn.lower(), idx_str):
        return True, False

    # we avoid case-invariant matching for some types of ontologies
    # rj update - unless they have a digit in them e.g. p27
    if (
        idx_str.startswith("ENS")
        or idx_str.startswith("CVCL")
        or idx_str.startswith("http://purl.obolibrary.org/obo/CLO_")
    ) and not any(regex.search(idx_str) for regex in PROBLEMATIC_GENE_NAMES_RE):
        return True, False

    # if it doesn't hit any other conditions, it should match
    # case-insensitively
    return False, True


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
