import logging
from collections import defaultdict
from typing import Iterable

from kazu.data.data import CuratedTermBehaviour, CuratedTerm
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_preprocessing.base import OntologyParser

logger = logging.getLogger(__name__)


def filter_curations_for_ner(
    curations: Iterable[CuratedTerm], parser: OntologyParser
) -> Iterable[CuratedTerm]:
    """Filter curations to retain those that can be used for dictionary based
    NER.

    Also checks the curations are represented in the internal database.
    """
    original_terms = defaultdict(set)
    inherited_terms = defaultdict(set)
    syn_db = SynonymDatabase()
    for curation in curations:
        if curation.behaviour is CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING:
            original_terms[curation.curated_synonym].add(curation)
        elif curation.behaviour is CuratedTermBehaviour.INHERIT_FROM_SOURCE_TERM:
            inherited_terms[curation.source_term].add(curation)
        else:
            # not an ner behaviour, so move on to the next curation in the loop
            continue
    for curated_synonym, curation_set in original_terms.items():
        if len(curation_set) > 1:
            # note, this shouldn't happen, as it should be handled by the curation validation logic.
            # however, just in case this logic changes, we'll leave this here
            logger.warning("multiple curations detected for %s, %s", curated_synonym, curation_set)
            term_norm = next(iter(curation_set)).term_norm_for_linking(parser.entity_class)
            if term_norm not in syn_db.get_all(parser.name):
                logger.warning(
                    "dictionary based NER needs an database entry for %s, %s, but none exists",
                    term_norm,
                    parser.name,
                )
                continue
        yield from curation_set
        yield from inherited_terms.pop(curated_synonym, set())
    if len(inherited_terms) > 0:
        logger.warning(
            "The following inherited curations were unmatched to a source term, and will be ignored for NER: %s",
            inherited_terms.values(),
        )
