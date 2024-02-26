import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast, Optional

import pandas as pd
from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTerm,
    SimpleValue,
    CuratedTerm,
    AssociatedIdSets,
    GlobalParserActions,
    IdsAndSource,
)
from kazu.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
)
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.autocuration import AutoCurator
from kazu.ontology_preprocessing.curation_utils import (
    CurationProcessor,
    CuratedTermConflictAnalyser,
    load_curated_terms,
    dump_curated_terms,
    CurationError,
)
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import (
    as_path,
    syn_terms_to_curations,
    PathLike,
)

logger = logging.getLogger(__name__)


#: The column name in a dataframe parsed with :meth:`~.OntologyParser.parse_to_dataframe`
#: for the column of the entity's default/preferred label
DEFAULT_LABEL = "default_label"
#: The column name for the id of each entity
IDX = "idx"
#: The column name for the synonyms/alternative labels for each entity
SYN = "syn"
#: The column name for the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology
MAPPING_TYPE = "mapping_type"
#: The origin of a dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc.
#: Note, this is different from the parser.name, as is used to identify the origin of a mapping back to a data source
DATA_ORIGIN = "data_origin"


_ONTOLOGY_UPGRADE_REPORT_DIR = "_ontology_upgrade_report"
_ONTOLOGY_DEFAULTS_FILENAME = "_defaults.jsonl"
_CURATION_REPORT_FILENAME = "_curation_report"


class OntologyParser(ABC):
    """Parse an ontology (or similar) into a set of outputs suitable for NLP entity
    linking.

    Implementations should have a class attribute 'name' to
    something suitably representative. The key method is
    :meth:`~.OntologyParser.parse_to_dataframe`, which should convert an
    input source to a dataframe suitable for further processing.

    The other important method is :meth:`~.find_kb`. This should parse an ID
    string (if required) and return the underlying source. This is
    important for composite resources that contain identifiers from
    different seed sources.

    See :ref:`ontology_parser` for a more detailed guide.

    Generally speaking, when parsing a data source, synonyms that are
    symbolic (as determined by the :class:`~.StringNormalizer`) that refer to more
    than one id are more likely to be ambiguous. Therefore, we assume
    they refer to unique concepts (e.g. COX 1 could be 'ENSG00000095303'
    OR 'ENSG00000198804', and thus they will yield multiple instances of
    :class:`~.EquivalentIdSet`. Non symbolic synonyms (i.e. noun phrases) are far
    less likely to refer to distinct entities, so we might want to merge
    the associated ID's non-symbolic ambiguous synonyms into a single
    :class:`~.EquivalentIdSet`. The result of :meth:`.StringNormalizer.classify_symbolic`
    forms the ``is_symbolic`` parameter to :meth:`~.score_and_group_ids`.

    If the underlying knowledgebase contains more than one entity type,
    muliple parsers should be implemented, subsetting accordingly (e.g.
    MEDDRA_DISEASE, MEDDRA_DIAGNOSTIC).
    """

    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns (note, IDX will become the index)
    minimum_metadata_column_names = [DEFAULT_LABEL, DATA_ORIGIN]

    def __init__(
        self,
        in_path: PathLike,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        autocurator: Optional[AutoCurator] = None,
        curations_path: Optional[PathLike] = None,
        global_actions: Optional[GlobalParserActions] = None,
        run_upgrade_report: bool = False,
        run_curation_report: bool = False,
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
            :class:`~.EquivalentIdSet`. See :meth:`~.OntologyParser.score_and_group_ids` for further details
        :param data_origin: The origin of this dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc. Note, this is different
            from the parser.name, as is used to identify the origin of a mapping back to a data source
        :param synonym_generator: optional CombinatorialSynonymGenerator. Used to generate synonyms for dictionary
            based NER matching
        :param autocurator: optional :class:`~.AutoCurator`. An AutoCurator contains a series of heuristics that
            determines what the default behaviour for a :class:`~.SynonymTerm` should be. For example, "Ignore
            any strings shorter than two characters or longer than 50 characters", or "use case sensitive matching when
            the SynonymTerm is symbolic"
        :param curations_path: path to jsonl file of :class:`~.CuratedTerm`\\s to override the defaults of the parser.
        :param global_actions: path to json file of :class:`~.GlobalParserActions` to apply to the parser.
        :param run_upgrade_report: Use when upgrading the version of the underlying data. When True, reports novel and
            obsolete terms in the model pack directory. Note that this will overwrite the default
            :class:`~.CuratedTerm`\\s associated with this parser in the model pack.
        :param run_curation_report: Use when adjusting the human curations. When True, creates a report in the model
            pack directory describing various aspects of the curation set, such as no-op curations,
            case sensitivity/mention confidence conflicts etc.
        """

        self.in_path = as_path(in_path)
        self.entity_class = entity_class
        self.name = name
        self.ontology_autocuration_set_path = self.in_path.parent.joinpath(
            f"{self.name}{_ONTOLOGY_DEFAULTS_FILENAME}"
        )
        if string_scorer is None:
            logger.warning(
                "no string scorer configured for %s. Synonym resolution disabled.", self.name
            )
        self.string_scorer = string_scorer
        self.synonym_merge_threshold = synonym_merge_threshold
        self.data_origin = data_origin
        self.synonym_generator = synonym_generator
        self.autocurator = autocurator
        self.curations_path = as_path(curations_path) if curations_path is not None else None
        self.global_actions = global_actions
        self.run_upgrade_report = run_upgrade_report
        self.run_curation_report = run_curation_report
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()

    @abstractmethod
    def find_kb(self, string: str) -> str:
        """Split an IDX somehow to find the ontology SOURCE reference.

        :param string: the IDX string to process
        :return:
        """
        pass

    def resolve_synonyms(self, synonym_df: pd.DataFrame) -> set[SynonymTerm]:

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
            mapping_type_set: frozenset[str] = frozenset(row[MAPPING_TYPE])
            syn_norm = row["syn_norm"]
            if len(syn_set) > 1:
                logger.debug("normaliser has merged %s into a single term: %s", syn_set, syn_norm)

            is_symbolic = all(
                StringNormalizer.classify_symbolic(x, self.entity_class) for x in syn_set
            )

            ids: set[str] = row[IDX]
            ids_and_source = set(
                (
                    idx,
                    self.find_kb(idx),
                )
                for idx in ids
            )
            associated_id_sets, agg_strategy = self.score_and_group_ids(ids_and_source, is_symbolic)

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
    ) -> tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """For a given data source, one normalised synonym may map to one or more id. In
        some cases, the ID may be duplicate/redundant (e.g. there are many chembl ids
        for paracetamol). In other cases, the ID may refer to distinct concepts (e.g.
        COX 1 could be 'ENSG00000095303' OR 'ENSG00000198804').

        Since synonyms from data sources are confused in such a manner, we need to decide some way to cluster them into
        a single :class:`~.SynonymTerm` concept, which in turn is a container for one or more :class:`~.EquivalentIdSet`
        (depending on whether the concept is ambiguous or not)

        The job of ``score_and_group_ids`` is to determine how many :class:`~.EquivalentIdSet`\\ s for a given set of
        ids should be produced.

        The default algorithm (which can be overridden by concrete parser implementations) works as follows:

        1. If no ``string_scorer`` is configured, create an :class:`~.EquivalentIdSet` for each id (strategy
           :attr:`~.EquivalentIdAggregationStrategy.NO_STRATEGY` - not recommended)
        2. If only one ID is referenced, or the associated normalised synonym string is not symbolic, group the
           ids into a single :class:`~.EquivalentIdSet` (strategy :attr:`~.EquivalentIdAggregationStrategy.UNAMBIGUOUS`)
        3. otherwise, compare the default label associated with each ID to every other default label. If it's above
           ``self.synonym_merge_threshold``, merge into one :class:`~.EquivalentIdSet`, if not, create a new one.

        recommendation: Use the :class:`~.SapbertStringSimilarityScorer` for comparison.

        .. important::
            Any calls to this method requires the metadata DB to be populated, as this is the store of
            :data:`~.DEFAULT_LABEL`.

        :param ids_and_source: ids to determine appropriate groupings of, and their associated sources
        :param is_symbolic: is the underlying synonym symbolic?
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

                DefaultLabels = set[str]
                id_list: list[tuple[IdsAndSource, DefaultLabels]] = []
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

    def _parse_df_if_not_already_parsed(self):
        if self.parsed_dataframe is None:
            self.parsed_dataframe = self.parse_to_dataframe()
            self.parsed_dataframe[DATA_ORIGIN] = self.data_origin
            self.parsed_dataframe[IDX] = self.parsed_dataframe[IDX].astype(str)
            # since we always need a value for DEFAULT_LABEL,
            # if the underlying data doesn't provide one, just use the IDX
            rows_without_default_label = self.parsed_dataframe.loc[
                pd.isnull(self.parsed_dataframe[DEFAULT_LABEL])
            ]
            rows_without_default_label[DEFAULT_LABEL] = rows_without_default_label[IDX]

    @kazu_disk_cache.memoize(ignore={0})
    def export_metadata(self, parser_name: str) -> dict[str, dict[str, SimpleValue]]:
        """Export the metadata from the ontology.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too
            much memory)
        :return: {idx:{metadata_key:metadata_value}}
        """
        self._parse_df_if_not_already_parsed()
        assert self.parsed_dataframe is not None
        metadata_columns = self.parsed_dataframe.columns
        metadata_columns = metadata_columns.drop([MAPPING_TYPE, SYN])
        metadata_df = self.parsed_dataframe[metadata_columns]
        metadata_df = metadata_df.drop_duplicates(subset=[IDX])
        metadata_df = metadata_df.dropna(
            axis=0, subset=["idx"] + OntologyParser.minimum_metadata_column_names
        )
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.minimum_metadata_column_names).issubset(metadata_df.columns)
        metadata = metadata_df.to_dict(orient="index")
        return cast(dict[str, dict[str, SimpleValue]], metadata)

    def process_curations(
        self, terms: set[SynonymTerm]
    ) -> tuple[Optional[list[CuratedTerm]], set[SynonymTerm]]:
        if not self.ontology_autocuration_set_path.exists() or self.run_upgrade_report:
            clean_curations = self.generate_clean_default_curations(
                terms, upgrade_report=self.run_upgrade_report
            )

        else:
            clean_curations = None

        clean_curations = self.build_curation_report(clean_curations)

        curation_processor = CurationProcessor(
            global_actions=self.global_actions,
            curations=list(clean_curations),
            parser_name=self.name,
            entity_class=self.entity_class,
            synonym_terms=terms,
        )
        return curation_processor.export_curations_and_final_terms()

    def build_curation_report(
        self, maybe_autocuration_set_clean: Optional[set[CuratedTerm]]
    ) -> set[CuratedTerm]:
        if maybe_autocuration_set_clean is None:
            autocuration_set_clean = load_curated_terms(self.ontology_autocuration_set_path)
        else:
            autocuration_set_clean = maybe_autocuration_set_clean
        if self.curations_path is None:
            logger.warning(
                "%s is configured to use raw ontology synonyms. This may result in noisy NER performance.",
                self.name,
            )
            return autocuration_set_clean

        elif self.curations_path.exists():
            logger.info(
                "%s curations file found",
                self.name,
            )
            human_curation_set = load_curated_terms(self.curations_path)
        else:
            raise RuntimeError(f"curations not found for {self.name} at {self.curations_path}")

        if not self.run_curation_report:
            curation_report_path = None
            human_curation_set_report_path = None
            human_and_autocuration_set_conflict_report_path = None
        else:
            curation_report_path = as_path(self.in_path).parent.joinpath(
                f"{self.name}{_CURATION_REPORT_FILENAME}"
            )
            if curation_report_path.exists():
                shutil.rmtree(curation_report_path)
            curation_report_path.mkdir()

            human_curation_set_report_path = curation_report_path.joinpath(
                "human_curation_conflict_report"
            )
            human_curation_set_report_path.mkdir()
            human_and_autocuration_set_conflict_report_path = curation_report_path.joinpath(
                "active_term_conflict_report"
            )
            human_and_autocuration_set_conflict_report_path.mkdir()

            logger.info(
                "%s reporting discrepancies in human curation set and autocuration set",
                self.name,
            )

        # set autofix to false so that issues are reported
        conflict_analyser = CuratedTermConflictAnalyser(self.entity_class, autofix=False)

        human_curation_report = conflict_analyser.verify_curation_set_integrity(
            human_curation_set, path=human_curation_set_report_path
        )

        if (
            len(human_curation_report.normalisation_conflicts) > 0
            or len(human_curation_report.case_conflicts) > 0
        ):
            raise CurationError(
                f"{self.name} conflicts detected in human curation set. Fix these before continuing (see "
                f"{human_curation_set_report_path})"
            )

        merged_set = conflict_analyser.merge_human_and_auto_curations(
            human_curations=human_curation_report.clean_curations,
            autocurations=autocuration_set_clean,
            path=curation_report_path,
        )

        human_and_autocuration_set_merged_curation_report = (
            conflict_analyser.verify_curation_set_integrity(
                merged_set, path=human_and_autocuration_set_conflict_report_path
            )
        )

        return human_and_autocuration_set_merged_curation_report.clean_curations

    def generate_clean_default_curations(
        self, terms: set[SynonymTerm], upgrade_report: bool = False
    ) -> set[CuratedTerm]:
        """

        :param terms:
        :param upgrade_report:
        :return:
        """
        logger.info(
            "%s clean default curation build triggered. This may take some time.",
            self.name,
        )

        new_version_autocuration_set = self._generate_dirty_default_curations(terms)

        # autofix is true, to ensure a clean set of curations
        conflict_analyser = CuratedTermConflictAnalyser(self.entity_class, autofix=True)
        new_version_autocuration_set_clean = conflict_analyser.verify_curation_set_integrity(
            new_version_autocuration_set
        ).clean_curations

        if upgrade_report:
            if not self.ontology_autocuration_set_path.exists():
                raise RuntimeError(
                    f"{self.name} previous version autocuration set not found when asked to build upgrade report, so comparison is not possible.",
                )
            else:
                upgrade_report_path = self._create_upgrade_report_path()
                previous_version_autocuration_set_clean = self._back_up_previous_curation_file(
                    upgrade_report_path
                )

        # note, this should run after we've loaded/backed up the previous version set, if required!
        logger.info(
            "%s updating autocuration set in model pack: %s",
            self.name,
            self.ontology_autocuration_set_path,
        )
        dump_curated_terms(
            new_version_autocuration_set_clean, self.ontology_autocuration_set_path, force=True
        )

        if upgrade_report:
            logger.info(
                "%s writing novel/obsolete terms after upgrade to %s",
                self.name,
                upgrade_report_path,
            )
            novel_set = new_version_autocuration_set_clean.difference(
                previous_version_autocuration_set_clean
            )
            if novel_set:
                dump_curated_terms(
                    novel_set,
                    upgrade_report_path.joinpath("novel_autocuration_terms.jsonl"),
                    force=True,
                )

                obsolete_set = previous_version_autocuration_set_clean.difference(novel_set)
                if obsolete_set:
                    dump_curated_terms(
                        obsolete_set,
                        upgrade_report_path.joinpath("obsolete_autocuration_terms.jsonl"),
                        force=True,
                    )

        return new_version_autocuration_set_clean

    def _create_upgrade_report_path(self) -> Path:
        upgrade_report_path = as_path(self.in_path).parent.joinpath(
            f"{self.name}{_ONTOLOGY_UPGRADE_REPORT_DIR}"
        )
        if upgrade_report_path.exists():
            shutil.rmtree(upgrade_report_path)

        upgrade_report_path.mkdir()
        return upgrade_report_path

    def _back_up_previous_curation_file(self, upgrade_report_path: Path) -> set[CuratedTerm]:
        logger.info(
            "%s loading previous version autocuration set",
            self.name,
        )
        previous_version_autocuration_set_clean = load_curated_terms(
            self.ontology_autocuration_set_path
        )
        backup_path = upgrade_report_path.joinpath(
            "old_" + self.ontology_autocuration_set_path.name
        )
        logger.info(
            "%s backing up previous version autocuration set to to %s",
            self.name,
            backup_path,
        )

        dump_curated_terms(previous_version_autocuration_set_clean, backup_path)
        return previous_version_autocuration_set_clean

    def _generate_dirty_default_curations(self, terms: set[SynonymTerm]) -> set[CuratedTerm]:
        """Dirty curations come directly from a set of :class:`.SynonymTerm`\\, and are
        optionally further modified by synonym generation and autocuration routines.

        They are not guaranteed to be conflict free - hence why they are 'dirty'.
        """
        default_term_set = syn_terms_to_curations(terms)
        if self.synonym_generator is not None:
            logger.info(
                "%s synonym generation configuration detected",
                self.name,
            )
            default_term_set = self.synonym_generator(default_term_set)
        if self.autocurator is not None:
            logger.info(
                "%s autocuration configuration detected",
                self.name,
            )
            default_term_set = set(self.autocurator(default_term_set))

        return default_term_set

    @kazu_disk_cache.memoize(ignore={0})
    def export_synonym_terms(self, parser_name: str) -> set[SynonymTerm]:
        """Export :class:`.SynonymTerm` from the parser.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too
            much memory)
        :return:
        """
        self._parse_df_if_not_already_parsed()
        assert self.parsed_dataframe is not None
        # ensure correct order
        syn_df = self.parsed_dataframe[self.all_synonym_column_names].copy()
        syn_df = syn_df.dropna(subset=[SYN])
        syn_df[SYN] = syn_df[SYN].apply(str.strip)
        syn_df.drop_duplicates(subset=self.all_synonym_column_names)
        assert set(OntologyParser.all_synonym_column_names).issubset(syn_df.columns)
        synonym_terms = self.resolve_synonyms(synonym_df=syn_df)
        return synonym_terms

    @kazu_disk_cache.memoize(ignore={0})
    def _populate_databases(
        self, parser_name: str
    ) -> tuple[Optional[list[CuratedTerm]], dict[str, dict[str, SimpleValue]], set[SynonymTerm]]:
        """Disk cacheable method that populates all databases.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too
            much memory)
        :return:
        """
        logger.info("populating database for %s from source", self.name)
        metadata = self.export_metadata(self.name)
        # metadata db needs to be populated before call to export_synonym_terms
        self.metadata_db.add_parser(self.name, self.entity_class, metadata)
        intermediate_synonym_terms = self.export_synonym_terms(self.name)
        maybe_ner_curations, final_syn_terms = self.process_curations(intermediate_synonym_terms)
        self.parsed_dataframe = None  # clear the reference to save memory

        self.synonym_db.add(self.name, final_syn_terms)
        return maybe_ner_curations, metadata, final_syn_terms

    def populate_databases(
        self, force: bool = False, return_curations: bool = False
    ) -> Optional[list[CuratedTerm]]:
        """Populate the databases with the results of the parser.

        Also calculates the term norms associated with any curations (if provided) which
        can then be used for Dictionary based NER

        :param force: do not use the cache for the ontology parser
        :param return_curations: should processed curations be returned?
        :return: curations if required
        """
        if self.name in self.synonym_db.loaded_parsers and not force and not return_curations:
            logger.debug("parser %s already loaded.", self.name)
            return None

        cache_key = self._populate_databases.__cache_key__(self, self.name)

        if force:
            kazu_disk_cache.delete(self.export_metadata.__cache_key__(self, self.name))
            kazu_disk_cache.delete(self.export_synonym_terms.__cache_key__(self, self.name))
            kazu_disk_cache.delete(cache_key)

        maybe_curations, metadata, final_syn_terms = self._populate_databases(self.name)

        if self.name not in self.synonym_db.loaded_parsers:
            logger.info("populating database for %s from cache", self.name)
            self.metadata_db.add_parser(self.name, self.entity_class, metadata)
            self.synonym_db.add(self.name, final_syn_terms)

        return maybe_curations if return_curations else None

    @abstractmethod
    def parse_to_dataframe(self) -> pd.DataFrame:
        """Implementations should override this method, returning a 'long, thin' :class:`pandas.DataFrame` of at least the following
        columns:


        [:data:`~.IDX`, :data:`~.DEFAULT_LABEL`, :data:`~.SYN`, :data:`~.MAPPING_TYPE`]

        | :data:`~.IDX`: the ontology id
        | :data:`~.DEFAULT_LABEL`: the preferred label
        | :data:`~.SYN`: a synonym of the concept
        | :data:`~.MAPPING_TYPE`: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology

        .. note:: It is the responsibility of the implementation of ``parse_to_dataframe`` to add default labels as synonyms.

        Any 'extra' columns will be added to the :class:`~kazu.database.in_memory_db.MetadataDatabase` as metadata fields for the
        given id in the relevant ontology.
        """
        pass
