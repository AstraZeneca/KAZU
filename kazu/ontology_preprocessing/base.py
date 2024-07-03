import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast, Optional

import pandas as pd
from kazu.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    LinkingCandidate,
    SimpleValue,
    OntologyStringResource,
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
    OntologyResourceProcessor,
    OntologyStringConflictAnalyser,
    load_ontology_string_resources,
    dump_ontology_string_resources,
    CurationError,
    AutofixStrategy,
    OntologyResourceSetConflictReport,
    OntologyResourceSetMergeReport,
    OntologyResourceSetCompleteReport,
)
from kazu.ontology_preprocessing.constants import IDX, DEFAULT_LABEL, DATA_ORIGIN, MAPPING_TYPE, SYN
from kazu.ontology_preprocessing.downloads import OntologyDownloader
from kazu.ontology_preprocessing.ontology_upgrade_report import OntologyUpgradeReport
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import (
    as_path,
    linking_candidates_to_ontology_string_resources,
    PathLike,
)

logger = logging.getLogger(__name__)


_ONTOLOGY_DEFAULTS_FILENAME = "_defaults.jsonl"
OntologyMetadata = dict[str, dict[str, SimpleValue]]


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
        ontology_downloader: Optional[OntologyDownloader] = None,
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
            determines what the default behaviour for a :class:`~.LinkingCandidate` should be. For example, "Ignore
            any strings shorter than two characters or longer than 50 characters", or "use case sensitive matching when
            the LinkingCandidate is symbolic"
        :param curations_path: path to jsonl file of human-curated :class:`~.OntologyStringResource`\\s to override the defaults of the parser.
        :param global_actions: path to json file of :class:`~.GlobalParserActions` to apply to the parser.
        :param ontology_downloader: optional :class:`~.OntologyDownloader` to download the ontology data from a remote source.
        """

        self.in_path = as_path(in_path)
        self.entity_class = entity_class
        self.name = name
        self.ontology_auto_generated_resources_set_path = self.in_path.parent.joinpath(
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
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()
        self.ontology_downloader = ontology_downloader

    @abstractmethod
    def find_kb(self, string: str) -> str:
        """Split an IDX somehow to find the ontology SOURCE reference.

        :param string: the IDX string to process
        :return:
        """
        pass

    def _resolve_candidates(self, candidates_df: pd.DataFrame) -> set[LinkingCandidate]:

        result = set()
        candidates_df["syn_norm"] = candidates_df[SYN].apply(
            StringNormalizer.normalize, entity_class=self.entity_class
        )

        for i, row in (
            candidates_df[["syn_norm", SYN, IDX, MAPPING_TYPE]]
            .groupby(["syn_norm"])
            .agg(set)
            .reset_index()
            .iterrows()
        ):

            syn_set = row[SYN]
            mapping_type_set: frozenset[str] = frozenset(row[MAPPING_TYPE])
            syn_norm = row["syn_norm"]
            if len(syn_set) > 1:
                logger.debug(
                    "normaliser has merged %s into a single candidate: %s", syn_set, syn_norm
                )

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

            result.add(
                LinkingCandidate(
                    synonym_norm=syn_norm,
                    raw_synonyms=frozenset(syn_set),
                    is_symbolic=is_symbolic,
                    mapping_types=mapping_type_set,
                    associated_id_sets=associated_id_sets,
                    parser_name=self.name,
                    aggregated_by=agg_strategy,
                )
            )

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
        a single :class:`~.LinkingCandidate` concept, which in turn is a container for one or more :class:`~.EquivalentIdSet`
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
            columns_that_should_never_be_empty = [
                IDX
            ] + OntologyParser.minimum_metadata_column_names
            missing_expected_values_mask = (
                self.parsed_dataframe[columns_that_should_never_be_empty].isna().any(axis="columns")
            )

            if missing_expected_values_mask.any():
                raise ValueError(
                    "The parsed dataframe for %s contains missing values expected for metadata. The relevant rows (i.e. entities) will be dropped and ignored: \n\n%s"
                    % self.name,
                    self.parsed_dataframe[missing_expected_values_mask].to_string(
                        max_rows=40, max_cols=10
                    ),
                )

            null_values_per_column = self.parsed_dataframe.isna().any(axis="index")
            columns_with_null_values = null_values_per_column[null_values_per_column].index
            if len(columns_with_null_values) > 0:
                logger.warning(
                    "Some metadata columns have empty values for %s. This is permitted, but if you write custom code to read 'extra' metadata columns"
                    " and depend on the property always being populated, you will likely get errors. Either ensure you can handle this field being 'None',"
                    " or clean the underlying ontology to ensure the value is always populated. Affected columns: \n\n%s",
                    self.name,
                    columns_with_null_values,
                )

    @kazu_disk_cache.memoize(ignore={0})
    def _export_metadata(self, parser_name: str) -> OntologyMetadata:
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
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.minimum_metadata_column_names).issubset(metadata_df.columns)
        metadata = metadata_df.to_dict(orient="index")
        return cast(OntologyMetadata, metadata)

    def _process_candidates_and_string_resources(
        self, candidates: set[LinkingCandidate], clean_resources: set[OntologyStringResource]
    ) -> tuple[Optional[list[OntologyStringResource]], set[LinkingCandidate]]:

        curation_processor = OntologyResourceProcessor(
            global_actions=self.global_actions,
            resources=list(clean_resources),
            parser_name=self.name,
            entity_class=self.entity_class,
            linking_candidates=candidates,
        )
        return curation_processor.export_resources_and_final_candidates()

    def _resolve_autogenerated_resources(
        self, candidates: set[LinkingCandidate]
    ) -> set[OntologyStringResource]:
        if not self.ontology_auto_generated_resources_set_path.exists():
            clean_resources = self._generate_clean_default_resources(candidates)

        else:
            clean_resources = load_ontology_string_resources(
                self.ontology_auto_generated_resources_set_path
            )
        return clean_resources

    def _create_human_conflict_report(self) -> OntologyResourceSetConflictReport:
        """Creates a report of resource conflicts originating from the human set."""

        assert self.curations_path is not None
        if self.curations_path.exists():
            logger.info(
                "%s curations file found",
                self.name,
            )
            human_curation_set = load_ontology_string_resources(self.curations_path)
        else:
            raise RuntimeError(f"curations not found for {self.name} at {self.curations_path}")

        # set autofix to None so that issues are reported
        conflict_analyser = OntologyStringConflictAnalyser(
            self.entity_class, autofix=AutofixStrategy.NONE
        )

        human_curation_report = conflict_analyser.verify_resource_set_integrity(human_curation_set)

        if len(human_curation_report.normalisation_conflicts) > 0:
            raise CurationError(
                f"{self.name} normalisation conflicts detected in human curation set. Fix these before continuing.\n\n{human_curation_report.normalisation_conflicts}"
            )

        if len(human_curation_report.case_conflicts) > 0:
            logger.warning(
                "%s case conflicts detected in human curation set for %s. These will not be used until they are fixed. Use the Kazu Resource Tool to fix these.",
                len(human_curation_report.case_conflicts),
                self.name,
            )
        return human_curation_report

    def _create_merge_report(
        self,
        human_curation_set: set[OntologyStringResource],
        auto_generated_resources_clean: set[OntologyStringResource],
    ) -> OntologyResourceSetMergeReport:
        """Creates a report of how a human resource set will affect an autogenerated
        resource set."""

        # set autofix to None so that issues are reported
        conflict_analyser = OntologyStringConflictAnalyser(
            self.entity_class, autofix=AutofixStrategy.NONE
        )

        return conflict_analyser.merge_human_and_auto_resources(
            human_curated_resources=human_curation_set,
            autocurated_resources=auto_generated_resources_clean,
        )

    def _create_combined_conflict_report(
        self, combined_resource_set: set[OntologyStringResource]
    ) -> OntologyResourceSetConflictReport:
        """Creates a report of resource conflicts originating from the combined human
        and autogenerated set."""
        # set autofix to None so that issues are reported
        conflict_analyser = OntologyStringConflictAnalyser(
            self.entity_class, autofix=AutofixStrategy.NONE
        )
        return conflict_analyser.verify_resource_set_integrity(combined_resource_set)

    def download_ontology(self) -> Path:
        """Download the ontology to the in_path.

        :return: Path of downloaded ontology.
        :raises: RuntimeError if no downloader is configured.
        """
        if self.ontology_downloader is None:
            raise RuntimeError(
                f"OntologyDownloader not configured for {self.name}",
            )
        logger.info("removing old ontology resources for %s at %s", self.name, self.in_path)
        self.ontology_downloader.delete_previous(self.in_path)
        logger.info("downloading new ontology resources for %s", self.name)
        return self.ontology_downloader.download(self.in_path)

    def upgrade_ontology_version(self) -> OntologyUpgradeReport:
        """Use when upgrading the version of the underlying ontology data, or when
        changing the configuration of the :class:`.AutoCurator`\\.

        Generate a report that describes the differences in generated :class:`~.OntologyStringResource`\\s
        between the two versions/configurations. Note that this depends on the existence
        of a set of autogenerated :class:`~.OntologyStringResource`\\s from the previous
        configuration.

        To use this method, simply replace the file/directory of the original ontology
        version with the new ontology version in the model pack.

        Note that calling this method will invalidate the disk cache.

        :return:
        """
        if not self.ontology_auto_generated_resources_set_path.exists():
            raise RuntimeError(
                f"{self.name} previous version autogenerated resources not found when asked to build upgrade report, so comparison is not possible.",
            )
        logger.info(
            "loading auto generated resources for previous version of %s",
            self.name,
        )
        original_autogenerated_resources = load_ontology_string_resources(
            self.ontology_auto_generated_resources_set_path
        )
        logger.info(
            "generating resources for new version of %s",
            self.name,
        )
        # need to invalidate cache here, otherwise the previous version of the linking candidates will be returned
        self.clear_cache()
        (
            intermediate_linking_candidates,
            _,
        ) = self._export_metadata_and_intermediate_linking_candidates()
        new_version_auto_generated_resources_clean = self._generate_clean_default_resources(
            intermediate_linking_candidates, save_resources=False
        )
        report = OntologyUpgradeReport(
            previous_version_auto_generated_resources_clean=original_autogenerated_resources,
            new_version_auto_generated_resources_clean=new_version_auto_generated_resources_clean,
        )
        # as a precaution, invalidate cache again in case pipeline is reinstantiated with new invalid cache
        self.clear_cache()
        return report

    def _generate_clean_default_resources(
        self, candidates: set[LinkingCandidate], save_resources: bool = True
    ) -> set[OntologyStringResource]:
        """From a set of candidates, produce default resources that do not conflict
        internally, based upon any configured autocuration regime.

        Note, they may still conflict with other parsers in your model pack.

        :param candidates:
        :param save_resources: should the generated resources be persisted within the
            model pack?
        :return:
        """
        logger.info(
            "%s clean default resources build triggered. This may take some time.",
            self.name,
        )
        # autofix is set to OPTIMISTIC, to ensure a clean set of resources
        conflict_analyser = OntologyStringConflictAnalyser(
            self.entity_class, autofix=AutofixStrategy.OPTIMISTIC
        )
        new_version_auto_generated_resources_dirty = self._generate_dirty_default_resources(
            candidates
        )
        new_version_auto_generated_resources_clean = (
            conflict_analyser.verify_resource_set_integrity(
                new_version_auto_generated_resources_dirty
            ).clean_resources
        )

        if save_resources:
            logger.info(
                "%s updating auto generated resources in model pack: %s",
                self.name,
                self.ontology_auto_generated_resources_set_path,
            )
            dump_ontology_string_resources(
                new_version_auto_generated_resources_clean,
                self.ontology_auto_generated_resources_set_path,
                force=True,
            )

        return new_version_auto_generated_resources_clean

    def _generate_dirty_default_resources(
        self, candidates: set[LinkingCandidate]
    ) -> set[OntologyStringResource]:
        """Dirty resources come directly from a set of :class:`.LinkingCandidate`\\, and
        are optionally further modified by synonym generation and autocuration routines.

        They are not guaranteed to be conflict free - hence why they are 'dirty'.
        """
        default_candidate_set = linking_candidates_to_ontology_string_resources(candidates)
        if self.synonym_generator is not None:
            logger.info(
                "%s synonym generation configuration detected",
                self.name,
            )
            default_candidate_set = self.synonym_generator(default_candidate_set)
        if self.autocurator is not None:
            logger.info(
                "%s autocuration configuration detected",
                self.name,
            )
            default_candidate_set = set(self.autocurator(default_candidate_set))

        return default_candidate_set

    @kazu_disk_cache.memoize(ignore={0})
    def _export_linking_candidates(self, parser_name: str) -> set[LinkingCandidate]:
        """Export :class:`.LinkingCandidate` from the parser.

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
        linking_candidates = self._resolve_candidates(candidates_df=syn_df)
        return linking_candidates

    @kazu_disk_cache.memoize(ignore={0})
    def _populate_databases(
        self, parser_name: str
    ) -> tuple[Optional[list[OntologyStringResource]], OntologyMetadata, set[LinkingCandidate],]:
        """Disk cacheable method that populates all databases.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too
            much memory)
        :return:
        """
        logger.info("populating database for %s from source", self.name)
        metadata, report = self.populate_metadata_db_and_resolve_string_resources()
        (
            maybe_ner_resources,
            final_linking_candidates,
        ) = self._process_candidates_and_string_resources(
            report.intermediate_linking_candidates, report.final_conflict_report.clean_resources
        )
        self.parsed_dataframe = None  # clear the reference to save memory

        self.synonym_db.add_parser(self.name, final_linking_candidates)
        return maybe_ner_resources, metadata, final_linking_candidates

    def populate_metadata_db_and_resolve_string_resources(
        self,
    ) -> tuple[OntologyMetadata, OntologyResourceSetCompleteReport]:
        """Loads the metadata DB and resolves any :class:`~.OntologyStringResource`\\s
        associated with this parser."""

        (
            intermediate_linking_candidates,
            metadata,
        ) = self._export_metadata_and_intermediate_linking_candidates()
        auto_generated_resources_clean = self._resolve_autogenerated_resources(
            intermediate_linking_candidates
        )
        if self.curations_path is None:
            logger.warning(
                "%s is configured to use raw ontology synonyms. This may result in noisy NER performance.",
                self.name,
            )
            return metadata, OntologyResourceSetCompleteReport(
                intermediate_linking_candidates=intermediate_linking_candidates,
                final_conflict_report=OntologyResourceSetConflictReport(
                    clean_resources=auto_generated_resources_clean,
                    merged_resources=set(),
                    normalisation_conflicts=set(),
                    case_conflicts=set(),
                ),
            )
        human_resource_report = self._create_human_conflict_report()

        merge_report = self._create_merge_report(
            human_curation_set=human_resource_report.clean_resources,
            auto_generated_resources_clean=auto_generated_resources_clean,
        )

        combined_report = self._create_combined_conflict_report(merge_report.effective_resources)

        return metadata, OntologyResourceSetCompleteReport(
            intermediate_linking_candidates=intermediate_linking_candidates,
            final_conflict_report=combined_report,
            human_conflict_report=human_resource_report,
            merge_report=merge_report,
        )

    def _export_metadata_and_intermediate_linking_candidates(
        self,
    ) -> tuple[set[LinkingCandidate], OntologyMetadata]:
        """Note that the metadata database loading must be handled here, as the call to
        ``self._export_linking_candidates`` may depend on it being loaded.

        :return:
        """
        metadata = self._export_metadata(self.name)
        # metadata db needs to be populated before call to export_linking_candidates
        self.metadata_db.add_parser(self.name, self.entity_class, metadata)
        intermediate_linking_candidates = self._export_linking_candidates(self.name)
        return intermediate_linking_candidates, metadata

    def populate_databases(
        self, force: bool = False, return_resources: bool = False
    ) -> Optional[list[OntologyStringResource]]:
        """Populate the databases with the results of the parser.

        Also calculates the synonym norms associated with any resources (if provided)
        which can then be used for Dictionary based NER

        :param force: do not use the cache for the ontology parser
        :param return_resources: should processed resources be returned?
        :return: resources if required
        """
        if self.name in self.synonym_db.loaded_parsers and not force and not return_resources:
            logger.debug("parser %s already loaded.", self.name)
            return None

        if force:
            self.clear_cache()

        maybe_resources, metadata, final_linking_candidates = self._populate_databases(self.name)

        if self.name not in self.synonym_db.loaded_parsers:
            logger.info("populating database for %s from cache", self.name)
            self.metadata_db.add_parser(self.name, self.entity_class, metadata)
            self.synonym_db.add_parser(self.name, final_linking_candidates)

        return maybe_resources if return_resources else None

    def clear_cache(self) -> None:
        """Clears the disk cache for this parser."""
        cache_key = self._populate_databases.__cache_key__(self, self.name)
        kazu_disk_cache.delete(self._export_metadata.__cache_key__(self, self.name))
        kazu_disk_cache.delete(self._export_linking_candidates.__cache_key__(self, self.name))
        kazu_disk_cache.delete(cache_key)

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
