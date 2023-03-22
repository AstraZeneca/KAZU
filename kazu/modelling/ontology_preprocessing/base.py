import functools
import json
import logging
import os
import re
import sqlite3
from abc import ABC
from functools import cache
from pathlib import Path
from typing import (
    cast,
    overload,
    List,
    Tuple,
    Dict,
    Any,
    Iterable,
    Set,
    Optional,
    FrozenSet,
    Union,
)
from urllib import parse

import pandas as pd
import rdflib

from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTerm,
    SimpleValue,
    Curation,
    ParserBehaviour,
    SynonymTermBehaviour,
    SynonymTermAction,
    AssociatedIdSets,
    GlobalParserActions,
)
from kazu.modelling.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    DBModificationResult,
)
from kazu.modelling.language.string_similarity_scorers import StringSimilarityScorer
from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import PathLike, as_path

# dataframe column keys
DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"
DATA_ORIGIN = "data_origin"
IdsAndSource = Set[Tuple[str, str]]

logger = logging.getLogger(__name__)


class CurationException(Exception):
    pass


@functools.cache
def load_curated_terms(
    path: PathLike,
) -> Optional[List[Curation]]:
    """
    Load :class:`kazu.data.data.Curation`\\ s from a file path.

    :param path: path to json lines file that map to :class:`kazu.data.data.Curation`
    :return:
    """
    curations_path = as_path(path)
    curations: Optional[List[Curation]] = None
    if curations_path.exists():
        with curations_path.open(mode="r") as jsonlf:
            curations = [Curation.from_json(json.loads(line)) for line in jsonlf]
    return curations


@functools.cache
def load_global_actions(
    path: PathLike,
) -> Optional[GlobalParserActions]:
    """
    Load an instance of GlobalParserActions  from a file path.

    :param path: path to a json serialised GlobalParserActions`
    :return:
    """
    global_actions_path = as_path(path)
    global_actions = None
    if global_actions_path.exists():
        with global_actions_path.open(mode="r") as jsonlf:
            global_actions = GlobalParserActions.from_json(json.load(jsonlf))
    return global_actions


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking.
    Implementations should have a class attribute 'name' to something suitably representative.
    The key method is parse_to_dataframe, which should convert an input source to a dataframe suitable
    for further processing.

    The other important method is find_kb. This should parse an ID string (if required) and return the underlying
    source. This is important for composite resources that contain identifiers from different seed sources

    Generally speaking, when parsing a data source, synonyms that are symbolic (as determined by
    the StringNormalizer) that refer to more than one id are more likely to be ambiguous. Therefore,
    we assume they refer to unique concepts (e.g. COX 1 could be 'ENSG00000095303' OR
    'ENSG00000198804', and thus they will yield multiple instances of EquivalentIdSet. Non symbolic
    synonyms (i.e. noun phrases) are far less likely to refer to distinct entities, so we might
    want to merge the associated ID's non-symbolic ambiguous synonyms into a single EquivalentIdSet.
    The result of StringNormalizer.is_symbolic forms the is_symbolic parameter to .score_and_group_ids.

    If the underlying knowledgebase contains more than one entity type, muliple parsers should be
    implemented, subsetting accordingly (e.g. MEDDRA_DISEASE, MEDDRA_DIAGNOSTIC).
    """

    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns (note, IDX will become the index)
    minimum_metadata_column_names = [DEFAULT_LABEL, DATA_ORIGIN]

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        """
        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param entity_class: The entity class to associate with this parser throughout the pipeline.
            Also used in the parser when calling StringNormalizer to determine the class-appropriate behaviour.
        :param name: A string to represent a parser in the overall pipeline. Should be globally unique
        :param string_scorer: Optional protocol of StringSimilarityScorer.  Used for resolving ambiguous symbolic
            synonyms via similarity calculation of the default label associated with the conflicted labels. If no
            instance is provided, all synonym conflicts will be assumed to refer to different concepts. This is not
            recommended!
        :param synonym_merge_threshold: similarity threshold to trigger a merge of conflicted synonyms into a single
            EquivalentIdSet. See docs for score_and_group_ids for further details
        :param data_origin: The origin of this dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc. Note, this is different
            from the parser.name, as is used to identify the origin of a mapping back to a data source
        :param synonym_generator: optional CombinatorialSynonymGenerator. Used to generate synonyms for dictionary
            based NER matching
        :param curations: Curations to apply to the parser
        """

        self.in_path = in_path
        self.entity_class = entity_class
        self.name = name
        if string_scorer is None:
            logger.warning(
                "no string scorer configured for %s. Synonym resolution disabled.", self.name
            )
        self.string_scorer = string_scorer
        self.synonym_merge_threshold = synonym_merge_threshold
        self.data_origin = data_origin
        self.synonym_generator = synonym_generator
        self.curations = curations
        self.global_actions = global_actions
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()

    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the ontology SOURCE reference

        :param string: the IDX string to process
        :return:
        """
        raise NotImplementedError()

    def resolve_synonyms(self, synonym_df: pd.DataFrame) -> Set[SynonymTerm]:

        result = set()
        synonym_df["syn_norm"] = synonym_df[SYN].apply(
            StringNormalizer.normalize, entity_class=self.entity_class
        )

        for i, row in (
            synonym_df[["syn_norm", SYN, IDX, MAPPING_TYPE]]
            .groupby(["syn_norm"])
            .agg(set)
            .reset_index()
            .iterrows()
        ):

            syn_set = row[SYN]
            mapping_type_set: FrozenSet[str] = frozenset(row[MAPPING_TYPE])
            syn_norm = row["syn_norm"]
            if len(syn_set) > 1:
                logger.debug(f"normaliser has merged {syn_set} into a single term: {syn_norm}")

            is_symbolic = all(
                StringNormalizer.classify_symbolic(x, self.entity_class) for x in syn_set
            )

            ids: Set[str] = row[IDX]
            ids_and_source = set(
                (
                    idx,
                    self.find_kb(idx),
                )
                for idx in ids
            )
            associated_id_sets, agg_strategy = self.score_and_group_ids(
                ids_and_source, is_symbolic, syn_set
            )

            synonym_term = SynonymTerm(
                term_norm=syn_norm,
                terms=frozenset(syn_set),
                is_symbolic=is_symbolic,
                mapping_types=mapping_type_set,
                associated_id_sets=associated_id_sets,
                parser_name=self.name,
                aggregated_by=agg_strategy,
            )

            result.add(synonym_term)

        return result

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """
        for a given data source, one normalised synonym may map to one or more id. In some cases, the ID may be
        duplicate/redundant (e.g. there are many chembl ids for paracetamol). In other cases, the ID may refer to
        distinct concepts (e.g. COX 1 could be 'ENSG00000095303' OR 'ENSG00000198804').


        Since synonyms from data sources are confused in such a manner, we need to decide some way to cluster them into
        a single SynonymTerm concept, which in turn is a container for one or more EquivalentIdSet (depending on
        whether the concept is ambiguous or not)

        The job of score_and_group_ids is to determine how many EquivalentIdSet's for a given set of ids should be
        produced.

        The default algorithm (which can be overridden by concrete parser implementations) works as follows:

        1. If no StringScorer is configured, create an EquivalentIdSet for each id (strategy NO_STRATEGY -
           not recommended)
        2. If only one ID is referenced, or the associated normalised synonym string is not symbolic, group the
           ids into a single EquivalentIdSet (strategy UNAMBIGUOUS)
        3. otherwise, compare the default label associated with each ID to every other default label. If it's above
           self.synonym_merge_threshold, merge into one EquivalentIdSet, if not, create a new one

        recommendation: Use the SapbertStringSimilarityScorer for comparison

        IMPORTANT NOTE: any calls to this method requires the metadata DB to be populated, as this is the store of
        DEFAULT_LABEL

        :param ids_and_source: ids to determine appropriate groupings of, and their associated sources
        :param is_symbolic: is the underlying synonym symbolic?
        :param original_syn_set: original synonyms associated with ids
        :return:
        """
        if self.string_scorer is None:
            # the NO_STRATEGY aggregation strategy assumes all synonyms are ambiguous
            return (
                frozenset(
                    EquivalentIdSet(ids_and_source=frozenset((single_id_and_source,)))
                    for single_id_and_source in ids_and_source
                ),
                EquivalentIdAggregationStrategy.NO_STRATEGY,
            )
        else:

            if len(ids_and_source) == 1:
                return (
                    frozenset((EquivalentIdSet(ids_and_source=frozenset(ids_and_source)),)),
                    EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                )

            if not is_symbolic:
                return (
                    frozenset((EquivalentIdSet(ids_and_source=frozenset(ids_and_source)),)),
                    EquivalentIdAggregationStrategy.MERGED_AS_NON_SYMBOLIC,
                )
            else:
                # use similarity to group ids into EquivalentIdSets

                DefaultLabels = Set[str]
                id_list: List[Tuple[IdsAndSource, DefaultLabels]] = []
                for id_and_source_tuple in ids_and_source:
                    default_label = cast(
                        str,
                        self.metadata_db.get_by_idx(self.name, id_and_source_tuple[0])[
                            DEFAULT_LABEL
                        ],
                    )
                    most_similar_id_set = None
                    best_score = 0.0
                    for id_and_default_label_set in id_list:
                        sim = max(
                            self.string_scorer(default_label, other_label)
                            for other_label in id_and_default_label_set[1]
                        )
                        if sim > self.synonym_merge_threshold and sim > best_score:
                            most_similar_id_set = id_and_default_label_set
                            best_score = sim

                    # for the first label, the above for loop is a no-op as id_sets is empty
                    # and the below if statement will be true.
                    # After that, it will be True if the id under consideration should not
                    # merge with any existing group and should get its own EquivalentIdSet
                    if not most_similar_id_set:
                        id_list.append(
                            (
                                {id_and_source_tuple},
                                {default_label},
                            )
                        )
                    else:
                        most_similar_id_set[0].add(id_and_source_tuple)
                        most_similar_id_set[1].add(default_label)

                return (
                    frozenset(
                        EquivalentIdSet(ids_and_source=frozenset(ids_and_source))
                        for ids_and_source, _ in id_list
                    ),
                    EquivalentIdAggregationStrategy.RESOLVED_BY_SIMILARITY,
                )

    def _attempt_to_add_database_entry_for_curation(
        self, id_set: Set[str], curated_synonym: str
    ) -> SynonymTerm:
        """
        Insert a new :class:`~kazu.data.data.SynonymTerm` into the database, or return an existing
        matching one if already present.



        :param id_set: if a :class:`.SynonymTerm` already exists for the normalised version of curated_synonym, this should be
            a subset of one of the :class:`.EquivalentIdSet`\\s assocaited with that term. If a :class:`.SynonymTerm` does not
            exist, the parser will attempt to find an existing instance of :class:`.EquivalentIdSet` that matches the ids in
            id_set. If no appropriate :class:`.EquivalentIdSet` exists, a new instance of :class:`kazu.data.data.AssociatedIdSets`
            will be created, containing an instance of :class:`.EquivalentIdSet` for each id in id_set
        :param curated_synonym: passed to :class:`.StringNormalizer` to see if a suitable :class:`.SynonymTerm` exists
        :return:
        """

        term_norm = StringNormalizer.normalize(curated_synonym, entity_class=self.entity_class)
        log_prefix = f"{self.name} attempting to create synonym term for <{curated_synonym}> term_norm: <{term_norm}> IDs: {id_set}"

        # look up the term norm in the db
        try:
            maybe_existing_synonym_term = self.synonym_db.get(self.name, term_norm)
        except KeyError:
            maybe_existing_synonym_term = None

        if maybe_existing_synonym_term is not None:
            all_ids_on_existing_syn_term = set(
                id_
                for equiv_id_set in maybe_existing_synonym_term.associated_id_sets
                for id_ in equiv_id_set.ids
            )
            if id_set.issubset(all_ids_on_existing_syn_term):
                logger.debug(
                    f"{log_prefix} but term_norm <{term_norm}> already exists in synonym database."
                    f"since this SynonymTerm includes all ids in id_set, no action is required. {maybe_existing_synonym_term.associated_id_sets}"
                )
                return maybe_existing_synonym_term
            else:
                raise CurationException(
                    f"{log_prefix} but term_norm <{term_norm}> already exists in synonym database, and its\n"
                    f"associated_id_sets don't contain all the ids in id_set. Creating a new\n"
                    f"SynonymTerm would override an existing entry, resulting in inconsistencies.\n"
                    f"This can happen if a synonym appears twice in the underlying ontology,\n"
                    f"with multiple identifiers attached\n"
                    f"Possible mitigations:\n"
                    f"1) use a ParserAction to drop the existing SynonymTerm from the database first.\n"
                    f"2) change the target id set of the curation to match the existing entry\n"
                    f"\t(i.e. {all_ids_on_existing_syn_term}\n"
                    f"3) Change the string normalizer function to generate unique term_norms\n"
                )

        logger.debug(
            f"no appropriate AssociatedIdSets exist for the set {id_set}, so a new one will be created"
        )
        # see if we've already had to group all the ids in this id_set in some way for a different synonym
        set_of_assoc_id_set = set()
        for idx in id_set:
            assoc_id_sets_for_this_id = self.synonym_db.get_associated_id_sets_for_id(
                self.name, idx
            )
            if len(assoc_id_sets_for_this_id) == 0:
                raise CurationException(
                    f"{log_prefix} but could not find element {idx} of id_set {id_set} in synonym database"
                )
            set_of_assoc_id_set.update(assoc_id_sets_for_this_id)

        associated_id_set_for_new_synonym_term = None

        # see if an existing AssociatedIdSet contains all the relevant IDs
        # we need the sort as we want to try to match to the smallest instance of AssociatedIdSets first.
        # This is because this is the least ambiguous - if we don't sort, we're potentially matching to
        # a larger, more ambiguous one than we need, and are potentially creating a disambiguation problem
        # where none exists
        for associated_id_set in sorted(set_of_assoc_id_set, key=len, reverse=False):
            all_ids_in_assoc_id_set = set(
                id_ for equiv_id_set in associated_id_set for id_ in equiv_id_set.ids
            )
            if id_set.issubset(all_ids_in_assoc_id_set):
                associated_id_set_for_new_synonym_term = associated_id_set
                logger.debug(
                    f"using smallest AssociatedIDSet that matches all IDs for new SynonymTerm: {associated_id_set}"
                )
                break

        if associated_id_set_for_new_synonym_term is None:
            # something to be careful about here: we assume that if no appropriate AssociatedIdSet can be
            # reused, we need to create a new one. If one cannot be found, we 'assume' that the input
            # id_sets must relate to different concepts (i.e. - we create a new equivalent ID set for each
            # id, which must later be disambiguated). This assumption may be inappropriate in cases. This is
            # best avoided by having the curation contain as few IDs as possible, such that the chances
            # that an existing AssociatedIdSet can be reused are higher.
            logger.debug(
                f"no appropriate AssociatedIdSets exist for the set {id_set}, so a new one will be created"
            )
            associated_id_set_for_new_synonym_term = frozenset(
                EquivalentIdSet(
                    ids_and_source=frozenset(
                        (
                            (
                                idx,
                                self.find_kb(idx),
                            ),
                        )
                    )
                )
                for idx in id_set
            )

        is_symbolic = StringNormalizer.classify_symbolic(curated_synonym, self.entity_class)
        new_term = SynonymTerm(
            term_norm=term_norm,
            terms=frozenset((curated_synonym,)),
            is_symbolic=is_symbolic,
            mapping_types=frozenset(("kazu_curated",)),
            associated_id_sets=associated_id_set_for_new_synonym_term,
            parser_name=self.name,
            aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
        )
        self.synonym_db.add(self.name, synonyms=(new_term,))
        logger.debug(f"{new_term} created")
        return new_term

    def _drop_synonym_term_for_linking(self, curated_synonym: str):
        """
        Remove a :class:`.SynonymTerm` from the database.

        :param curated_synonym: passed to :class:`.StringNormalizer` to look up the :class:`.SynonymTerm`
        :return:
        """
        affected_term_key = StringNormalizer.normalize(
            curated_synonym, entity_class=self.entity_class
        )
        try:
            self.synonym_db.drop_synonym_term(self.name, affected_term_key)
            logger.debug(
                "successfully dropped %s from database for %s", affected_term_key, self.name
            )
        except KeyError:
            logger.warning(
                "tried to drop %s from database, but key doesn't exist for %s",
                affected_term_key,
                self.name,
            )

    def _drop_id_set_from_synonym_term(self, id_set: Set[str], curated_synonym: str):
        """
        Remove an id set from a :class:`.SynonymTerm`.

        :param id_set: ids that should be removed from the :class:`.SynonymTerm`
        :param curated_synonym: passed to :class:`.StringNormalizer` to look up the :class:`.SynonymTerm`
        :return:
        """
        # make a mutable copy so we can discard as we go
        mutable_id_set = set(id_set)
        affected_term_key = StringNormalizer.normalize(
            curated_synonym, entity_class=self.entity_class
        )
        target_term_to_modify = self.synonym_db.get(self.name, affected_term_key)
        for equiv_id_set in target_term_to_modify.associated_id_sets:
            if len(mutable_id_set) == 0:
                break

            if len(mutable_id_set.intersection(equiv_id_set.ids)) > 0:
                drop_equivalent_id_set_from_synonym_term_result = (
                    self.synonym_db.drop_equivalent_id_set_from_synonym_term(
                        self.name, affected_term_key, equiv_id_set
                    )
                )
                if (
                    drop_equivalent_id_set_from_synonym_term_result
                    is DBModificationResult.ID_SET_MODIFIED
                ):
                    logger.debug(
                        "dropped an EquivalentIdSet containing an id from %s for key %s for %s",
                        id_set,
                        affected_term_key,
                        self.name,
                    )
                else:
                    logger.debug(
                        "dropped a SynonymTerm containing an id from %s for key %s for %s",
                        id_set,
                        affected_term_key,
                        self.name,
                    )
                mutable_id_set.difference_update(equiv_id_set.ids)
        else:
            logger.warning(
                "Was asked to remove ids associated with a SynonymTerm (key: <%s>). However, after inspecting all"
                " EquivalentIdSets, the following ids were not found in any of them: %s. Parser name: %s",
                affected_term_key,
                mutable_id_set,
                self.name,
            )

    def process_actions(self) -> Optional[List[Curation]]:
        """
        Process any global actions or curations associated with this parser.

        :return: curations that are suitable for dictionary based NER for this parser.
        """
        if self.global_actions is not None:
            ids_dropped_through_global_actions = self._process_global_actions(self.global_actions)
        else:
            ids_dropped_through_global_actions = set()
        if self.curations is not None:
            curation_with_term_norm_actions = []
            for curation in self.curations:
                maybe_curation_with_term_norm_actions = self._process_curation(
                    curation, ids_dropped_through_global_actions
                )
                if maybe_curation_with_term_norm_actions is not None:
                    curation_with_term_norm_actions.append(maybe_curation_with_term_norm_actions)
            return curation_with_term_norm_actions
        else:
            logger.info("No curations provided for %s", self.name)
            return None

    def _update_action_for_globally_dropped_ids(
        self,
        curation_id: Dict[str, str],
        action: SynonymTermAction,
        ids_dropped_through_global_actions: Set[str],
    ) -> Optional[SynonymTermAction]:
        """
        Checks the action to see if it's id has been dropped by a global action elsewhere. If so, it's
        modified accordingly and returned. If the action will no longer work after modification, None is
        returned.

        :param curation_id:
        :param action:
        :param ids_dropped_through_global_actions:
        :return:
        """
        original_ids = action.parser_to_target_id_mappings[self.name]

        filtered_ids = original_ids.difference(ids_dropped_through_global_actions)
        if len(filtered_ids) == 0:
            logger.warning(
                "curation id %s has had all linking target ids removed by a global action, and will be"
                " ignored. Parser name: %s",
                curation_id,
                self.name,
            )
            return None
        if len(filtered_ids) < len(original_ids):
            logger.warning(
                "curation found with ids that have been removed via a global action. These will be filtered"
                "from the curation action. Parser name: %s, new ids: %s, curation id: %s",
                self.name,
                filtered_ids,
                curation_id,
            )
            action.parser_to_target_id_mappings[self.name] = filtered_ids

        return action

    def _process_curation(
        self, curation: Curation, ids_dropped_through_global_actions: Set[str]
    ) -> Optional[Curation]:
        """
        Handle any parser specific behaviour associated with a :class:`.Curation`\\.

        :param curation:
        :param ids_dropped_through_global_actions:
        :return:
        """
        maybe_updated_curation = None
        for action in curation.parser_behaviour(self.name):
            if action.behaviour is SynonymTermBehaviour.IGNORE:
                logger.debug("ignoring unwanted curation: %s for %s", curation, self.name)
            elif action.behaviour is SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING:
                self._drop_synonym_term_for_linking(curated_synonym=curation.curated_synonym)
            elif action.behaviour is SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM:
                self._drop_id_set_from_synonym_term(
                    action.parser_to_target_id_mappings[self.name],
                    curated_synonym=curation.curated_synonym,
                )
            elif (
                action.behaviour is SynonymTermBehaviour.ADD_FOR_LINKING_ONLY
                or action.behaviour is SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING
            ):
                updated_action = self._update_action_for_globally_dropped_ids(
                    curation._id, action, ids_dropped_through_global_actions
                )
                if updated_action is not None:
                    new_or_existing_term = self._attempt_to_add_database_entry_for_curation(
                        action.parser_to_target_id_mappings[self.name],
                        curated_synonym=curation.curated_synonym,
                    )
                    if action.behaviour is SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING:
                        action.term_norm = new_or_existing_term.term_norm
                        maybe_updated_curation = curation
            else:
                raise ValueError(f"unknown behaviour for parser {self.name}, {action}")

        return maybe_updated_curation

    def _process_global_actions(self, global_actions: GlobalParserActions) -> Set[str]:
        """
        Process global actions associated with this parser, returning a set of any ids
        that have been dropped.

        :param global_actions:
        :return:
        """
        dropped_ids = set()
        for action in global_actions.parser_behaviour(self.name):
            if action.behaviour is ParserBehaviour.DROP_IDS_FROM_PARSER:
                ids = action.parser_to_target_id_mappings[self.name]
                terms_modified, terms_dropped = 0, 0
                for idx in ids:
                    terms_modified_this_id, terms_dropped_this_id = self.synonym_db.drop_id_from_all_synonym_terms(self.name, idx)  # type: ignore[arg-type]
                    terms_modified += terms_modified_this_id
                    terms_dropped += terms_dropped_this_id
                    if terms_modified_this_id == 0 and terms_dropped_this_id == 0:
                        logger.warning("failed to drop %s from %s", idx, self.name)
                    else:
                        dropped_ids.add(idx)
                        logger.debug(
                            "dropped ID %s from %s. SynonymTerm modified count: %s, SynonymTerm dropped count: %s",
                            idx,
                            self.name,
                            terms_modified_this_id,
                            terms_dropped_this_id,
                        )
            else:
                raise ValueError(f"unknown behaviour for parser {self.name}, {action}")
        return dropped_ids

    def _parse_df_if_not_already_parsed(self):
        if self.parsed_dataframe is None:
            self.parsed_dataframe = self.parse_to_dataframe()
            self.parsed_dataframe[DATA_ORIGIN] = self.data_origin
            self.parsed_dataframe[IDX] = self.parsed_dataframe[IDX].astype(str)
            self.parsed_dataframe.loc[
                pd.isnull(self.parsed_dataframe[DEFAULT_LABEL]), DEFAULT_LABEL
            ] = self.parsed_dataframe[IDX]

    def export_metadata(self) -> Dict[str, Dict[str, SimpleValue]]:
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        metadata_columns = self.parsed_dataframe.columns
        metadata_columns.drop([MAPPING_TYPE, SYN])
        metadata_df = self.parsed_dataframe[metadata_columns]
        metadata_df = metadata_df.drop_duplicates(subset=[IDX]).dropna(axis=0)
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.minimum_metadata_column_names).issubset(metadata_df.columns)
        metadata = metadata_df.to_dict(orient="index")
        return cast(Dict[str, Dict[str, SimpleValue]], metadata)

    def export_synonym_terms(self) -> Set[SynonymTerm]:
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        # ensure correct order
        syn_df = self.parsed_dataframe[self.all_synonym_column_names].copy()
        syn_df = syn_df.dropna(subset=[SYN])
        syn_df[SYN] = syn_df[SYN].apply(str.strip)
        syn_df.drop_duplicates(subset=self.all_synonym_column_names)
        assert set(OntologyParser.all_synonym_column_names).issubset(syn_df.columns)
        synonym_terms = self.resolve_synonyms(synonym_df=syn_df)
        return synonym_terms

    def populate_metadata_database(self):
        """
        populate the metadata database with this ontology
        """
        self.metadata_db.add_parser(self.name, self.export_metadata())

    def generate_synonyms(self) -> Set[SynonymTerm]:
        """
        Generate synonyms based on configured synonym generator.

        Note, this method also calls populate_databases(), as the metadata db must be populated
        for appropriate synonym resolution.
        """
        self.populate_databases()
        synonym_data = set(self.synonym_db.get_all(self.name).values())
        generated_synonym_data = set()
        if self.synonym_generator:
            generated_synonym_data = self.synonym_generator(synonym_data)
        generated_synonym_data.update(synonym_data)
        logger.info(
            f"{len(synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return generated_synonym_data

    def populate_synonym_database(self):
        """
        populate the synonym database
        """

        self.synonym_db.add(self.name, self.export_synonym_terms())

    def populate_databases(self, force: bool = False) -> Optional[List[Curation]]:
        """
        populate the databases with the results of the parser. Also calculates the term norms associated with
        any curations (if provided) which can then be used for Dictionary based NER

        :param force: normally, this call does nothing if databases already have an entry for this parser.
            this can be forced by setting this param to True
        :return: curations with term norms
        """

        if self.name in self.synonym_db.loaded_parsers and not force:
            logger.info("will not repopulate databases as already populated for %s", self.name)
            return None
        else:
            self.populate_metadata_database()
            self.populate_synonym_database()
            curations_with_term_norms = self.process_actions()
            self.parsed_dataframe = None  # clear the reference to save memory
            return curations_with_term_norms

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of at least the following
        columns:


        [IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE]

        IDX: the ontology id
        DEFAULT_LABEL: the preferred label
        SYN: a synonym of the concept
        MAPPING_TYPE: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology

        Note: It is the responsibility of the implementation of parse_to_dataframe to add default labels as synonyms.

        Any 'extra' columns will be added to the :class:`~kazu.modelling.database.in_memory_db.MetadataDatabase` as metadata fields for the
        given id in the relevant ontology.
        """
        raise NotImplementedError()


class JsonLinesOntologyParser(OntologyParser):
    """
    A parser for a jsonlines dataset. Assumes one kb entry per line (i.e. json object)
    implemetations should implement json_dict_to_parser_dict (see method notes for details
    """

    def read(self, path: str) -> Iterable[Dict[str, Any]]:
        for json_path in Path(path).glob("*.json"):
            with json_path.open(mode="r") as f:
                for line in f:
                    yield json.loads(line)

    def parse_to_dataframe(self):
        return pd.DataFrame.from_records(self.json_dict_to_parser_records(self.read(self.in_path)))

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        """
        for a given input json (represented as a python dict), yield dictionary record(s) compatible with the expected
        structure of the Ontology Parser superclass - i.e. should have keys for SYN, MAPPING_TYPE, DEFAULT_LABEL and
        IDX. All other keys are used as mapping metadata

        :param jsons_gen: iterator of python dict representing json objects
        :return:
        """
        raise NotImplementedError()


class OpenTargetsDiseaseOntologyParser(JsonLinesOntologyParser):
    # Just use IDs that are in MONDO, since that's all people in general care about.
    # if we want to expand this out, other sources are:
    # "OGMS", "FBbt", "Orphanet", "EFO", "OTAR"
    # but we did have these in previously, and EFO introduced a lot of noise as it
    # has non-disease terms like 'dose' that occur frequently.
    # we could make the allowed sources a config option but we don't need to configure
    # currently, and easy to change later (and provide the current value as a default if
    # not present in config)
    allowed_sources = {"MONDO", "HP"}

    def find_kb(self, string: str) -> str:
        return string.split("_")[0]

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        # we ignore related syns for now until we decide how the system should handle them
        for json_dict in jsons_gen:
            idx = self.look_for_mondo(json_dict["id"], json_dict.get("dbXRefs", []))
            if any(allowed_source in idx for allowed_source in self.allowed_sources):
                synonyms = json_dict.get("synonyms", {})
                exact_syns = synonyms.get("hasExactSynonym", [])
                exact_syns.append(json_dict["name"])
                def_label = json_dict["name"]
                dbXRefs = json_dict.get("dbXRefs", [])
                for syn in exact_syns:
                    yield {
                        SYN: syn,
                        MAPPING_TYPE: "hasExactSynonym",
                        DEFAULT_LABEL: def_label,
                        IDX: idx,
                        "dbXRefs": dbXRefs,
                    }

    def look_for_mondo(self, ot_id: str, db_xrefs: List[str]):
        if "MONDO" in ot_id:
            return ot_id
        for x in db_xrefs:
            if "MONDO" in x:
                return x.replace(":", "_")
        return ot_id


class OpenTargetsTargetOntologyParser(JsonLinesOntologyParser):

    annotation_fields = {
        "subcellularLocations",
        "tractability",
        "constraint",
        "functionDescriptions",
        "go",
        "hallmarks",
        "chemicalProbes",
        "safetyLiabilities",
        "pathways",
        "targetClass",
    }

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """
        since non symbolic gene symbols are also frequently ambiguous, we override this method accordingly to disable
        all synonym resolution, and rely on disambiguation to decide on 'true' mappings. Answers on a postcard if anyone
        has a better idea on how to do this!

        :param id_and_source:
        :param is_symbolic:
        :param original_syn_set:
        :return:
        """

        return (
            frozenset(
                EquivalentIdSet(ids_and_source=frozenset((single_id_and_source,)))
                for single_id_and_source in ids_and_source
            ),
            EquivalentIdAggregationStrategy.CUSTOM,
        )

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        for json_dict in jsons_gen:
            # due to a bug in OT data, TEC genes have "gene" as a synonym. Sunce they're uninteresting, we just filter
            # them
            biotype = json_dict.get("biotype")
            if biotype == "" or biotype == "tec" or json_dict["id"] == json_dict["approvedSymbol"]:
                continue

            annotation_score = sum(
                1
                for annotation_field in self.annotation_fields
                if len(json_dict.get(annotation_field, [])) > 0
            )

            shared_values = {
                IDX: json_dict["id"],
                DEFAULT_LABEL: json_dict["approvedSymbol"],
                "dbXRefs": json_dict.get("dbXRefs", []),
                "approvedName": json_dict["approvedName"],
                "annotation_score": annotation_score,
            }

            for key in ["synonyms", "obsoleteSymbols", "obsoleteNames", "proteinIds"]:
                synonyms_and_sources_lst = json_dict.get(key, [])
                for record in synonyms_and_sources_lst:
                    if "label" in record and "id" in record:
                        raise RuntimeError(f"record: {record} has both id and label specified")
                    elif "label" in record:
                        record[SYN] = record.pop("label")
                    elif "id" in record:
                        record[SYN] = record.pop("id")
                    record[MAPPING_TYPE] = record.pop("source")
                    record.update(shared_values)
                    yield record

            for key in ("approvedSymbol", "approvedName", "id"):
                if key == "id":
                    mapping_type = "opentargets_id"
                else:
                    mapping_type = key

                res = {SYN: json_dict[key], MAPPING_TYPE: mapping_type}
                res.update(shared_values)
                yield res


class OpenTargetsMoleculeOntologyParser(JsonLinesOntologyParser):
    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def json_dict_to_parser_records(
        self, jsons_gen: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        for json_dict in jsons_gen:
            cross_references = json_dict.get("crossReferences", {})
            default_label = json_dict["name"]
            idx = json_dict["id"]

            synonyms = json_dict.get("synonyms", [])
            main_name = json_dict["name"]
            synonyms.append(main_name)

            for syn in synonyms:
                yield {
                    SYN: syn,
                    MAPPING_TYPE: "synonyms",
                    "crossReferences": cross_references,
                    DEFAULT_LABEL: default_label,
                    IDX: idx,
                }

            for trade_name in json_dict.get("tradeNames", []):
                yield {
                    SYN: trade_name,
                    MAPPING_TYPE: "tradeNames",
                    "crossReferences": cross_references,
                    DEFAULT_LABEL: default_label,
                    IDX: idx,
                }


RdfRef = Union[rdflib.paths.Path, rdflib.term.Node, str]
# Note - lists are actually normally provided here through hydra config
# but there's apparently no way of type hinting
# 'any iterable of length two where the items have these types'
PredicateAndValue = Tuple[RdfRef, rdflib.term.Node]


class RDFGraphParser(OntologyParser):
    """
    Parser for Owl files.
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        uri_regex: Union[str, re.Pattern],
        synonym_predicates: Iterable[RdfRef],
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        exclude_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
        )

        if isinstance(uri_regex, re.Pattern):
            self._uri_regex = uri_regex
        else:
            self._uri_regex = re.compile(uri_regex)

        self.synonym_predicates = tuple(
            self.convert_to_rdflib_ref(pred) for pred in synonym_predicates
        )

        if include_entity_patterns is not None:
            self.include_entity_patterns = tuple(
                (self.convert_to_rdflib_ref(pred), self.convert_to_rdflib_ref(val))
                for pred, val in include_entity_patterns
            )
        else:
            self.include_entity_patterns = tuple()

        if exclude_entity_patterns is not None:
            self.exclude_entity_patterns = tuple(
                (self.convert_to_rdflib_ref(pred), self.convert_to_rdflib_ref(val))
                for pred, val in exclude_entity_patterns
            )
        else:
            self.exclude_entity_patterns = tuple()

    def find_kb(self, string: str) -> str:
        # By default, just return the name of the parser.
        # If more complex behaviour is necessary, write a custom subclass and override this method.
        return self.name

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: rdflib.paths.Path) -> rdflib.paths.Path:
        ...

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: rdflib.term.Node) -> rdflib.term.Node:
        ...

    @overload
    @staticmethod
    def convert_to_rdflib_ref(pred: str) -> rdflib.URIRef:
        ...

    @staticmethod
    def convert_to_rdflib_ref(pred):
        if isinstance(pred, (rdflib.term.Node, rdflib.paths.Path)):
            return pred
        else:
            return rdflib.URIRef(pred)

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = rdflib.Graph()
        g.parse(self.in_path)
        label_pred_str = "http://www.w3.org/2000/01/rdf-schema#label"
        label_predicates = rdflib.URIRef(label_pred_str)
        default_labels = []
        iris = []
        syns = []
        mapping_type = []

        for sub, obj in g.subject_objects(label_predicates):
            if not self.is_valid_iri(str(sub)):
                continue

            # type ignore is necessary because rdflib's typing thinks that for Graph.__contains__ can't use an rdflib.paths.Path
            # as a predicate, but you can, because __contains__ calls Graph.triples(), which is type hinted to allow Paths (and
            # reading the implementation it clearly handles Paths).
            if any((sub, pred, value) not in g for pred, value in self.include_entity_patterns):  # type: ignore[operator]
                continue

            # as above
            if any((sub, pred, value) in g for pred, value in self.exclude_entity_patterns):  # type: ignore[operator]
                continue

            default_labels.append(str(obj))
            iris.append(str(sub))
            syns.append(str(obj))
            mapping_type.append(label_pred_str)
            for syn_predicate in self.synonym_predicates:
                for other_syn_obj in g.objects(subject=sub, predicate=syn_predicate):
                    default_labels.append(str(obj))
                    iris.append(str(sub))
                    syns.append(str(other_syn_obj))
                    mapping_type.append(str(syn_predicate))

        df = pd.DataFrame.from_dict(
            {DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns, MAPPING_TYPE: mapping_type}
        )
        return df

    def is_valid_iri(self, text: str) -> bool:
        """
        Check if input string is a valid IRI for the ontology being parsed.

        Uses `self._uri_regex` to define valid IRIs
        """
        match = self._uri_regex.match(text)
        return bool(match)


class GeneOntologyParser(OntologyParser):
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/GO_[0-9]+$")

    instances: Set[str] = set()
    instances_in_dbs: Set[str] = set()

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        query: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
        )
        self.instances.add(name)
        self.query = query

    def populate_databases(self, force: bool = False) -> Optional[List[Curation]]:
        curations_with_term_norms = super().populate_databases(force=force)
        self.instances_in_dbs.add(self.name)

        if self.instances_in_dbs >= self.instances:
            # all existing instances are in the database, so we can free up
            # the memory used by the cached parsed gene ontology, which is significant.
            self.load_go.cache_clear()
        return curations_with_term_norms

    def __del__(self):
        GeneOntologyParser.instances.discard(self.name)

    @staticmethod
    @cache
    def load_go(in_path: PathLike) -> rdflib.Graph:
        g = rdflib.Graph()
        g.parse(in_path)
        return g

    def find_kb(self, string: str) -> str:
        return self.name

    def parse_to_dataframe(self) -> pd.DataFrame:
        g = self.load_go(self.in_path)
        result = g.query(self.query)
        default_labels = []
        iris = []
        syns = []
        mapping_type = []

        # there seems to be a bug in rdflib that means the iterator sometimes exits early unless we covert to list first
        # type cast is necessary because iterating over an rdflib query result gives different types depending on the kind
        # of query, so rdflib gives a Union here, but we know it should be a ResultRow because we know we should have a
        # select query
        list_res = cast(List[rdflib.query.ResultRow], list(result))
        for row in list_res:
            idx = str(row.goid)
            label = str(row.label)
            if "obsolete" in label:
                logger.debug(f"skipping obsolete id: {idx}, {label}")
                continue
            if self._uri_regex.match(idx):
                default_labels.append(label)
                iris.append(idx)
                syns.append(str(row.synonym))
                mapping_type.append("hasExactSynonym")
        df = pd.DataFrame.from_dict(
            {DEFAULT_LABEL: default_labels, IDX: iris, SYN: syns, MAPPING_TYPE: mapping_type}
        )
        default_labels_df = df[[IDX, DEFAULT_LABEL]].drop_duplicates().copy()
        default_labels_df[SYN] = default_labels_df[DEFAULT_LABEL]
        default_labels_df[MAPPING_TYPE] = "label"

        return pd.concat([df, default_labels_df])


class BiologicalProcessGeneOntologyParser(GeneOntologyParser):
    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations=curations,
            global_actions=global_actions,
            query="""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                    SELECT DISTINCT ?goid ?label ?synonym
                            WHERE {

                                ?goid oboinowl:hasExactSynonym ?synonym .
                                ?goid rdfs:label ?label .
                                ?goid oboinowl:hasOBONamespace "biological_process" .

                  }
            """,
        )


class MolecularFunctionGeneOntologyParser(GeneOntologyParser):
    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations=curations,
            global_actions=global_actions,
            query="""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                    SELECT DISTINCT ?goid ?label ?synonym
                            WHERE {

                                ?goid oboinowl:hasExactSynonym ?synonym .
                                ?goid rdfs:label ?label .
                                ?goid oboinowl:hasOBONamespace "molecular_function".

                    }
            """,
        )


class CellularComponentGeneOntologyParser(GeneOntologyParser):
    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations=curations,
            global_actions=global_actions,
            query="""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX oboinowl: <http://www.geneontology.org/formats/oboInOwl#>

                    SELECT DISTINCT ?goid ?label ?synonym
                            WHERE {

                                ?goid oboinowl:hasExactSynonym ?synonym .
                                ?goid rdfs:label ?label .
                                ?goid oboinowl:hasOBONamespace "cellular_component" .

                    }
            """,
        )


class UberonOntologyParser(RDFGraphParser):
    """
    input should be an UBERON owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/uberon
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/UBERON_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
        )

    def find_kb(self, string: str) -> str:
        return "UBERON"


class MondoOntologyParser(OntologyParser):
    _uri_regex = re.compile("^http://purl.obolibrary.org/obo/(MONDO|HP)_[0-9]+$")
    """
    input should be a MONDO json file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/mondo
    """

    def find_kb(self, string: str):
        path = parse.urlparse(string).path
        # just the final bit, e.g. MONDO_0000123
        path_end = path.split("/")[-1]
        # we don't want the underscore or digits for the unique ID, just the ontology bit
        return path_end.split("_")[0]

    def parse_to_dataframe(self) -> pd.DataFrame:
        x = json.load(open(self.in_path, "r"))
        graph = x["graphs"][0]
        nodes = graph["nodes"]
        ids = []
        default_label_list = []
        all_syns = []
        mapping_type = []
        for i, node in enumerate(nodes):
            if not self.is_valid_iri(node["id"]):
                continue

            idx = node["id"]
            default_label = node.get("lbl")
            if default_label is None:
                # skip if no default label is available
                continue
            # add default_label to syn type
            all_syns.append(default_label)
            default_label_list.append(default_label)
            mapping_type.append("lbl")
            ids.append(idx)

            syns = node.get("meta", {}).get("synonyms", [])
            for syn_dict in syns:

                pred = syn_dict["pred"]
                if pred in {"hasExactSynonym"}:
                    mapping_type.append(pred)
                    syn = syn_dict["val"]
                    ids.append(idx)
                    default_label_list.append(default_label)
                    all_syns.append(syn)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label_list, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df

    def is_valid_iri(self, text: str) -> bool:
        match = self._uri_regex.match(text)
        return bool(match)


class EnsemblOntologyParser(OntologyParser):
    """
    input is a json from HGNC
    e.g. http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json

    :return:
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            name=name,
            curations=curations,
            global_actions=global_actions,
        )

    def find_kb(self, string: str) -> str:
        return "ENSEMBL"

    def parse_to_dataframe(self) -> pd.DataFrame:

        keys_to_check = [
            "name",
            "symbol",
            "uniprot_ids",
            "alias_name",
            "alias_symbol",
            "prev_name",
            "lncipedia",
            "prev_symbol",
            "vega_id",
            "refseq_accession",
            "hgnc_id",
            "mgd_id",
            "rgd_id",
            "ccds_id",
            "pseudogene.org",
        ]

        with open(self.in_path, "r") as f:
            data = json.load(f)
        ids = []
        default_label = []
        all_syns = []
        all_mapping_type: List[str] = []
        docs = data["response"]["docs"]
        for doc in docs:

            def get_with_default_list(key: str):
                found = doc.get(key, [])
                if not isinstance(found, list):
                    found = [found]
                return found

            ensembl_gene_id = doc.get("ensembl_gene_id", None)
            name = doc.get("name", None)
            if ensembl_gene_id is None or name is None:
                continue
            else:
                # find synonyms
                synonyms: List[Tuple[str, str]] = []
                for hgnc_key in keys_to_check:
                    synonyms_this_entity = get_with_default_list(hgnc_key)
                    for potential_synonym in synonyms_this_entity:
                        synonyms.append((potential_synonym, hgnc_key))

                synonyms = list(set(synonyms))
                synonyms_strings = []
                for synonym_str, mapping_t in synonyms:
                    all_mapping_type.append(mapping_t)
                    synonyms_strings.append(synonym_str)

                num_syns = len(synonyms_strings)
                ids.extend([ensembl_gene_id] * num_syns)
                default_label.extend([name] * num_syns)
                all_syns.extend(synonyms_strings)

        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_label, SYN: all_syns, MAPPING_TYPE: all_mapping_type}
        )
        return df


class ChemblOntologyParser(OntologyParser):
    """
    input is a sqllite dump from Chembl, e.g.
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29_sqlite.tar.gz
    """

    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def parse_to_dataframe(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.in_path)
        query = f"""\
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN}, syn_type AS {MAPPING_TYPE}
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN}, "pref_name" AS {MAPPING_TYPE}
            FROM molecule_dictionary
        """
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])

        df.drop_duplicates(inplace=True)

        return df


class CLOOntologyParser(RDFGraphParser):
    """
    input is a CLO Owl file
    https://www.ebi.ac.uk/ols/ontologies/clo
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/CLO_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
        )

    def find_kb(self, string: str) -> str:
        return "CLO"


class CellosaurusOntologyParser(OntologyParser):
    """
    input is an obo file from cellosaurus, e.g.
    https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo
    """

    cell_line_re = re.compile("cell line", re.IGNORECASE)

    def find_kb(self, string: str) -> str:
        return "CELLOSAURUS"

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """
        treat all synonyms as seperate cell lines

        :param ids:
        :param id_to_source:
        :param is_symbolic:
        :param original_syn_set:
        :return:
        """

        return (
            frozenset(
                EquivalentIdSet(
                    ids_and_source=frozenset((single_id_and_source,)),
                )
                for single_id_and_source in ids_and_source
            ),
            EquivalentIdAggregationStrategy.CUSTOM,
        )

    def _remove_cell_line_text(self, text: str):
        return self.cell_line_re.sub("", text).strip()

    def parse_to_dataframe(self) -> pd.DataFrame:

        ids = []
        default_labels = []
        all_syns = []
        mapping_type = []
        with open(self.in_path, "r") as f:
            id = ""
            for line in f:
                text = line.rstrip()
                if text.startswith("id:"):
                    id = text.split(" ")[1]
                elif text.startswith("name:"):
                    default_label = text[5:].strip()
                    ids.append(id)
                    # we remove "cell line" because they're all cell lines and it confuses mapping
                    default_label_no_cell_line = self._remove_cell_line_text(default_label)
                    default_labels.append(default_label_no_cell_line)
                    all_syns.append(default_label_no_cell_line)
                    mapping_type.append("name")
                # synonyms in cellosaurus are a bit of a mess, so we don't use this field for now. Leaving this here
                # in case they improve at some point
                # elif text.startswith("synonym:"):
                #     match = self._synonym_regex.match(text)
                #     if match is None:
                #         raise ValueError(
                #             """synonym line does not match our synonym regex.
                #             Either something is wrong with the file, or it has updated
                #             and our regex is not correct/general enough."""
                #         )
                #     ids.append(id)
                #     default_labels.append(default_label)
                #
                #     all_syns.append(self._remove_cell_line_text(match.group("syn")))
                #     mapping_type.append(match.group("mapping"))
                else:
                    pass
        df = pd.DataFrame.from_dict(
            {IDX: ids, DEFAULT_LABEL: default_labels, SYN: all_syns, MAPPING_TYPE: mapping_type}
        )
        return df

    _synonym_regex = re.compile(
        r"""^synonym:      # line that begins synonyms
        \s*                # any amount of whitespace (standardly a single space)
        "(?P<syn>[^"]*)"   # a quoted string - capture this as a named match group 'syn'
        \s*                # any amount of separating whitespace (standardly a single space)
        (?P<mapping>\w*)   # a sequence of word characters representing the mapping type
        \s*                # any amount of separating whitespace (standardly a single space)
        \[\]               # an open and close bracket at the end of the string
        $""",
        re.VERBOSE,
    )


class MeddraOntologyParser(OntologyParser):
    """
    input is an unzipped directory to a Meddra release (Note, requires licence). This
    should contain the files 'mdhier.asc' and 'llt.asc'.
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
        exclude_socs: Iterable[str] = (
            "Surgical and medical procedures",
            "Social circumstances",
            "Investigations",
        ),
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
            name=name,
        )

        self.exclude_socs = exclude_socs

    _mdhier_asc_col_names = (
        "pt_code",
        "hlt_code",
        "hlgt_code",
        "soc_code",
        "pt_name",
        "hlt_name",
        "hlgt_name",
        "soc_name",
        "soc_abbrev",
        "null_field",
        "pt_soc_code",
        "primary_soc_fg",
        "NULL",
    )

    _llt_asc_column_names = (
        "llt_code",
        "llt_name",
        "pt_code",
        "llt_whoart_code",
        "llt_harts_code",
        "llt_costart_sym",
        "llt_icd9_code",
        "llt_icd9cm_code",
        "llt_icd10_code",
        "llt_currency",
        "llt_jart_code",
        "NULL",
    )

    def find_kb(self, string: str) -> str:
        return "MEDDRA"

    def parse_to_dataframe(self) -> pd.DataFrame:
        # hierarchy path
        mdheir_path = os.path.join(self.in_path, "mdhier.asc")
        # low level term path
        llt_path = os.path.join(self.in_path, "llt.asc")
        # note: this isn't true! But the pandas stubs currently think that the names argument
        # here has to be a list of str, but it doesn't, it just has to be a sequence of strings
        list_mdhier_names = cast(List[str], self._mdhier_asc_col_names)
        hier_df = pd.read_csv(
            mdheir_path,
            sep="$",
            header=None,
            names=list_mdhier_names,
            dtype="string",
        )
        hier_df = hier_df[~hier_df["soc_name"].isin(self.exclude_socs)]

        # as above
        list_llt_names = cast(List[str], self._llt_asc_column_names)
        llt_df = pd.read_csv(
            llt_path,
            sep="$",
            header=None,
            names=list_llt_names,
            usecols=("llt_name", "pt_code"),
            dtype="string",
        )
        llt_df = llt_df.dropna(axis=1)

        ids = []
        default_labels = []
        all_syns = []
        mapping_type = []
        soc_names = []
        soc_codes = []

        for i, row in hier_df.iterrows():
            idx = row["pt_code"]
            pt_name = row["pt_name"]
            soc_name = row["soc_name"]
            soc_code = row["soc_code"]
            llts = llt_df[llt_df["pt_code"] == idx]
            ids.append(idx)
            default_labels.append(pt_name)
            all_syns.append(pt_name)
            soc_names.append(soc_name)
            soc_codes.append(soc_code)
            mapping_type.append("meddra_link")
            for j, llt_row in llts.iterrows():
                ids.append(idx)
                default_labels.append(pt_name)
                soc_names.append(soc_name)
                soc_codes.append(soc_code)
                all_syns.append(llt_row["llt_name"])
                mapping_type.append("meddra_link")

        for i, row in (
            hier_df[["hlt_code", "hlt_name", "soc_name", "soc_code"]].drop_duplicates().iterrows()
        ):
            ids.append(row["hlt_code"])
            default_labels.append(row["hlt_name"])
            soc_names.append(row["soc_name"])
            soc_codes.append(row["soc_code"])
            all_syns.append(row["hlt_name"])
            mapping_type.append("meddra_link")
        for i, row in (
            hier_df[["hlgt_code", "hlgt_name", "soc_name", "soc_code"]].drop_duplicates().iterrows()
        ):
            ids.append(row["hlgt_code"])
            default_labels.append(row["hlgt_name"])
            soc_names.append(row["soc_name"])
            soc_codes.append(row["soc_code"])
            all_syns.append(row["hlgt_name"])
            mapping_type.append("meddra_link")
        df = pd.DataFrame.from_dict(
            {
                IDX: ids,
                DEFAULT_LABEL: default_labels,
                SYN: all_syns,
                MAPPING_TYPE: mapping_type,
                "soc_name": soc_names,
                "soc_code": soc_codes,
            }
        )
        return df


class CLOntologyParser(RDFGraphParser):
    """
    input should be an CL owl file
    e.g.
    https://www.ebi.ac.uk/ols/ontologies/cl
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        exclude_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/CL_[0-9]+$"),
            synonym_predicates=(
                rdflib.URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            ),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations=curations,
            global_actions=global_actions,
        )

    def find_kb(self, string: str) -> str:
        return "CL"


class HGNCGeneFamilyParser(OntologyParser):

    syn_column_keys = {"Family alias", "Common root gene symbol"}

    def find_kb(self, string: str) -> str:
        return "HGNC_GENE_FAMILY"

    def parse_to_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.in_path, sep="\t")
        data = []
        for family_id, row in (
            df.groupby(by="Family ID").agg(lambda col_series: set(col_series.dropna())).iterrows()
        ):
            # in theory, there should only be one family name per ID
            assert len(row["Family name"]) == 1
            default_label = next(iter(row["Family name"]))
            data.append(
                {
                    SYN: default_label,
                    MAPPING_TYPE: "Family name",
                    DEFAULT_LABEL: default_label,
                    IDX: family_id,
                }
            )
            data.extend(
                {
                    SYN: syn,
                    MAPPING_TYPE: key,
                    DEFAULT_LABEL: default_label,
                    IDX: family_id,
                }
                for key in self.syn_column_keys
                for syn in row[key]
            )
        return pd.DataFrame.from_records(data)


class TabularOntologyParser(OntologyParser):
    """
    For already tabulated data.
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
        **kwargs,
    ):
        """

        :param in_path:
        :param entity_class:
        :param name:
        :param string_scorer:
        :param synonym_merge_threshold:
        :param data_origin:
        :param synonym_generator:
        :param curations:
        :param kwargs: passed to pandas.read_csv
        """
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
        )
        self._raw_dataframe = pd.read_csv(self.in_path, **kwargs)

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        Assume input file is already in correct format.

        Inherit and override this method if different behaviour is required.

        :return:
        """
        return self._raw_dataframe

    def find_kb(self, string: str) -> str:
        return self.name


class ATCDrugClassificationParser(TabularOntologyParser):
    """
    Parser for the ATC Drug classification dataset.

    This requires a licence from WHO, available at
    https://www.who.int/tools/atc-ddd-toolkit/atc-classification .

    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            curations=curations,
            global_actions=global_actions,
            sep="     ",
            header=None,
            names=["code", "level_and_description"],
            # Because the c engine can't handle multi-char sep
            # removing this results in the same behaviour, but
            # pandas logs a warning.
            engine="python",
        )

    levels_to_ignore = {"1", "2", "3"}

    def parse_to_dataframe(self) -> pd.DataFrame:
        # for some reason, the level and description codes are merged, so we need to fix this here
        df = self._raw_dataframe.applymap(str.strip)
        res_df = pd.DataFrame()
        res_df[[MAPPING_TYPE, DEFAULT_LABEL]] = df.apply(
            lambda row: [row["level_and_description"][0], row["level_and_description"][1:]],
            axis=1,
            result_type="expand",
        )
        res_df[IDX] = df["code"]
        res_df = res_df[~res_df[MAPPING_TYPE].isin(self.levels_to_ignore)]
        res_df[SYN] = res_df[DEFAULT_LABEL]
        return res_df


class StatoParser(RDFGraphParser):
    """
    Parse stato: input should be an owl file.

    Available at e.g.
    https://www.ebi.ac.uk/ols/ontologies/stato .
    """

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.7,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        include_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        exclude_entity_patterns: Optional[Iterable[PredicateAndValue]] = None,
        curations: Optional[List[Curation]] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):

        super().__init__(
            in_path=in_path,
            entity_class=entity_class,
            name=name,
            uri_regex=re.compile("^http://purl.obolibrary.org/obo/(OBI|STATO)_[0-9]+$"),
            synonym_predicates=(rdflib.URIRef("http://purl.obolibrary.org/obo/IAO_0000111"),),
            string_scorer=string_scorer,
            synonym_merge_threshold=synonym_merge_threshold,
            data_origin=data_origin,
            synonym_generator=synonym_generator,
            include_entity_patterns=include_entity_patterns,
            exclude_entity_patterns=exclude_entity_patterns,
            curations=curations,
            global_actions=global_actions,
        )

    def find_kb(self, string: str) -> str:
        return "OBI" if "OBI" in string else "STATO"
