import dataclasses
from collections import defaultdict
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import open_dict

from kazu.data.data import CuratedTerm
from kazu.utils.constants import HYDRA_VERSION_BASE

_SOURCE_TERMS_PREFIX = "source_terms"
_GENERATED_TERMS_PREFIX = "generated_terms"
_MIGRATED_TERMS_FILENAME = "_migrated_terms.jsonl"
_EXTRA_ONTOLOGY_TERMS_FILENAME = "_extra_ontology_terms.jsonl"
_MODIFIED_TERMS_FILENAME = "_modified_terms.jsonl"
_CASE_WARNING_TERMS_FILENAME = "_matched_terms_case_warnings.jsonl"
_OBSOLETE_TERMS_FILENAME = "_obsolete_terms.jsonl"
_NOVEL_TERMS_FILENAME = "_novel_terms.jsonl"

CURATION_REPORT_INSTRUCTIONS = f"""The subdirectories contain various jsonl files of how the current set of CuratedTerms map to the configured ontology.
{_MIGRATED_TERMS_FILENAME}: terms that are compatible with the configured version of the ontology
{_MODIFIED_TERMS_FILENAME}: terms that match but only if case is not considered. Since only one set of behavioursa exist between the original CuratedTerm and the new match, these CuratedTerms have been migrated to the new match.
{_CASE_WARNING_TERMS_FILENAME}: terms that match if case is not considered, but with multiple candidates in the new Ontology. These need to be recurated.
{_OBSOLETE_TERMS_FILENAME}: terms from the set of CuratedTerms that are now obsolete and can be removed.
{_NOVEL_TERMS_FILENAME}: terms from the Ontology that are novel and need to be curated.
{_EXTRA_ONTOLOGY_TERMS_FILENAME}: manually added terms that don't exist in the ontology, or are mapped to different id's than the ontology default

To summarise what to do:

recurate the following files:
{_SOURCE_TERMS_PREFIX}{_CASE_WARNING_TERMS_FILENAME}
{_SOURCE_TERMS_PREFIX}{_NOVEL_TERMS_FILENAME}
{_GENERATED_TERMS_PREFIX}{_CASE_WARNING_TERMS_FILENAME}
{_GENERATED_TERMS_PREFIX}{_NOVEL_TERMS_FILENAME}

then run

rm  *{_OBSOLETE_TERMS_FILENAME}

cat *.jsonl > """


class _OntologyUpgradeReport:
    def __init__(
        self,
        existing_curations: set[CuratedTerm],
        curations_from_new_ontology_version: set[CuratedTerm],
    ):

        self.incoming_curations_case_sensitive: defaultdict[str, set[CuratedTerm]] = defaultdict(
            set
        )
        self.incoming_curations_case_insensitive: defaultdict[str, set[CuratedTerm]] = defaultdict(
            set
        )
        for curation in curations_from_new_ontology_version:
            self.incoming_curations_case_sensitive[curation.curated_synonym].add(curation)
            self.incoming_curations_case_insensitive[curation.curated_synonym.lower()].add(curation)

        self.existing_curations_case_sensitive: defaultdict[str, set[CuratedTerm]] = defaultdict(
            set
        )
        for curation in existing_curations:
            self.existing_curations_case_sensitive[curation.curated_synonym].add(curation)

        # old/new curations that perfectly match on .curated_synonym
        self.matched_curations: set[CuratedTerm] = set()

        # a pointer to new curations that can be discarded
        self.eliminated_generated_curations_case_insensitive: defaultdict[
            str, set[CuratedTerm]
        ] = defaultdict(set)

        # curations that match in a case insensitive fashion, and can be automatically modified
        self.modified_curations: set[CuratedTerm] = set()

        # terms that don't appear in the ontology, but have been manually added
        self.extra_ontology_terms: set[CuratedTerm] = set()

        # terms that match only if case not considered
        self.failed_migrations: defaultdict[str, set[CuratedTerm]] = defaultdict(set)

        # terms that no longer appear in the ontology
        self.obsolete_terms: defaultdict[str, set[CuratedTerm]] = defaultdict(set)

        # terms that are new in the latest version of the ontology
        self.novel_terms: defaultdict[str, set[CuratedTerm]] = defaultdict(set)

    def match_existing_terms_to_new_terms(
        self,
    ) -> None:

        for existing_term_string_case_sensitive in list(
            self.existing_curations_case_sensitive.keys()
        ):
            existing_terms = self.existing_curations_case_sensitive[
                existing_term_string_case_sensitive
            ]
            matched_generated_terms = self.incoming_curations_case_sensitive.pop(
                existing_term_string_case_sensitive, None
            )
            if matched_generated_terms is not None:
                # we were able to find a match, so this set of terms is still good
                self.matched_curations.update(existing_terms)
                # pop this key as it's already been handled and we don't need to worry about it any more
                self.existing_curations_case_sensitive.pop(existing_term_string_case_sensitive)
                # keep track of the generated terms for this string, so we know that the case sensitive versions have been
                # handled. We'll need this later when we look at case insensitive matching
                self.eliminated_generated_curations_case_insensitive[
                    existing_term_string_case_sensitive.lower()
                ].update(matched_generated_terms)

        # now loop again over the remaining items, considering the more difficult cases where exact matching wasn't possible
        # we need to loop twice as we want to ensure every case sensitive match first, to ensure self.eliminated_generated_curations_ci
        # is fully populated and we don't miss any curations that could be mapped.
        for (
            existing_term_string_case_sensitive,
            existing_terms,
        ) in self.existing_curations_case_sensitive.items():
            matched_generated_terms_ci = self.incoming_curations_case_insensitive.pop(
                existing_term_string_case_sensitive.lower(), None
            )
            if matched_generated_terms_ci is None:
                # if no match, term is either extra ontological (i.e. added separately) or obsolete
                for term in existing_terms:
                    if term.additional_to_source:
                        self.extra_ontology_terms.add(term)
                    elif (
                        term.source_term is None
                    ):  # we group by source term for easy reviewing later
                        self.obsolete_terms[term.curated_synonym].add(term)
                    else:
                        self.obsolete_terms[term.source_term].add(term)
            else:
                # if there's a case insensitive match, remove any curations that have already been handled by case sensitive matching
                matched_generated_terms_ci_minus_already_handled_ones = (
                    matched_generated_terms_ci.difference(
                        self.eliminated_generated_curations_case_insensitive[
                            existing_term_string_case_sensitive.lower()
                        ]
                    )
                )
                # we can only migrate a term if there is only a single set of behaviours from the original curation
                human_behaviour_set = set(x.control_aspects for x in existing_terms)
                if len(human_behaviour_set) == 1:
                    human_term = next(iter(existing_terms))
                    for generated_term in matched_generated_terms_ci_minus_already_handled_ones:
                        self.modified_curations.add(
                            dataclasses.replace(
                                generated_term,
                                associated_id_sets=human_term.associated_id_sets,
                                behaviour=human_term.behaviour,
                                case_sensitive=human_term.case_sensitive,
                            )
                        )
                else:
                    # add to failure list, so the generated terms can be curated
                    for generated_term in matched_generated_terms_ci_minus_already_handled_ones:
                        self.failed_migrations[generated_term.curated_synonym].add(generated_term)

        # any remaining terms are novel
        for (
            generated_term_string_case_insensitive,
            matched_generated_terms_ci,
        ) in self.incoming_curations_case_insensitive.items():
            matched_generated_terms_ci_minus_already_handled_ones = (
                matched_generated_terms_ci.difference(
                    self.eliminated_generated_curations_case_insensitive[
                        generated_term_string_case_insensitive.lower()
                    ]
                )
            )
            for term in matched_generated_terms_ci_minus_already_handled_ones:
                if term.source_term is None:
                    self.novel_terms[term.curated_synonym].add(term)
                else:
                    self.novel_terms[term.source_term].add(term)

    def write_curation_report(
        self, output_path: Path, parser_name: str, prefix: str, curation_file_name: str
    ) -> None:
        results_dir = Path(output_path).joinpath("curation_reports").joinpath(parser_name)
        results_dir.mkdir(exist_ok=True, parents=True)
        with results_dir.joinpath(f"{prefix}{_MIGRATED_TERMS_FILENAME}").open(mode="w") as f:
            for curation in self.matched_curations:
                f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath(f"{prefix}{_EXTRA_ONTOLOGY_TERMS_FILENAME}").open(mode="w") as f:
            for curation in self.extra_ontology_terms:
                f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath(f"{prefix}{_MODIFIED_TERMS_FILENAME}").open(mode="w") as f:
            for curation in self.modified_curations:
                f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath(f"{prefix}{_CASE_WARNING_TERMS_FILENAME}").open(mode="w") as f:
            by_len = sorted(self.failed_migrations.keys(), key=lambda x: len(x), reverse=False)
            for term_str in by_len:
                terms = self.failed_migrations[term_str]
                for curation in terms:
                    f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath(f"{prefix}{_OBSOLETE_TERMS_FILENAME}").open(mode="w") as f:
            by_len = sorted(self.obsolete_terms.keys(), key=lambda x: len(x), reverse=False)
            for term_str in by_len:
                terms = self.obsolete_terms[term_str]
                for curation in terms:
                    f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath(f"{prefix}{_NOVEL_TERMS_FILENAME}").open(mode="w") as f:
            by_len = sorted(self.novel_terms.keys(), key=lambda x: len(x), reverse=False)
            for term_str in by_len:
                terms = self.novel_terms[term_str]
                for curation in terms:
                    f.write(f"{curation.to_json()}\n")
        with results_dir.joinpath("instructions.txt").open(mode="w") as f:
            f.write(CURATION_REPORT_INSTRUCTIONS + curation_file_name)


def run_curation_report(model_pack_path: Path) -> None:
    """Assists with upgrading CuratedTerms to a new version of an Ontology.

    .. warning::

       Calling this function will invalidate the kazu disk cache in the model pack.

    Since the currency of NER is strings, but the currency of entity
    linking is identifiers, it can be challenging to migrate an existing
    set of CuratedTerms to a new version of an ontology. Many things can
    change, such as terms become obsolete, their case may change, and/or
    new terms can be introduced.

    All the while, one wants to be able to reuse as many of the human
    curations one has made as possible.

    The aim of this class is to assist with this process as far as is
    possible, by producing a report that migrates previous curations,
    modifies existing curations to match their equivalents in a new
    version of an ontology, and highlights obsolete/new terms for
    curation.

    The algorithm runs as follows:

    1. Run synonym generation routines on the new version of the ontology to produce a set of :class:`.CuratedTerm`\\.

    2. Match generated synonyms with existing curations on a case-sensitive basis.

    3. Per synonym, if a match is detected, keep the original :class:`.CuratedTerm` and disregard any generated
       :class:`.CuratedTerm`\\s.

    4. For any remaining existing curations, match again in a case-insensitive fashion.

    5. If no match is detected, the term is either manually added, or is now obsolete in the original ontology.

    6. If a match is detected, the term may be migrated. This means copying the behaviour of the original
       :class:`.CuratedTerm` to the newly generated term. This is only possible if a single set of control aspects
       is defined for this synonym (defined by :attr:`.CuratedTerm.control_aspects`\\. Otherwise, the term
       will have to be manually curated again.

    7. If no match is detected, the newly generated :class:`.CuratedTerm` is novel to this version of the ontology and
       must be curated.

    A curation report directory will be created in the model pack, with subdirectories per parser. Follow on instructions
    can be found in these directories.
    """

    # relative imports so function can be imported without initialising disk cache
    from kazu.utils.caching import kazu_disk_cache
    from kazu.ontology_preprocessing.base import load_curated_terms

    with initialize_config_dir(
        version_base=HYDRA_VERSION_BASE, config_dir=str(model_pack_path.joinpath("conf"))
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "hydra/job_logging=none",
                "hydra/hydra_logging=none",
            ],
        )
        for parser_name in cfg["ontologies"]["parsers"]:
            curations_path = Path(cfg["ontologies"]["parsers"][parser_name]["curations_path"])

            existing_source_term_curations = set()
            existing_generated_curations = set()
            for human_curated_term in load_curated_terms(curations_path):
                if human_curated_term.source_term is None:
                    existing_source_term_curations.add(human_curated_term)
                else:
                    existing_generated_curations.add(human_curated_term)

            new_source_term_curations = set()
            new_generated_curations = set()
            with open_dict(cfg):
                cfg["ontologies"]["parsers"][parser_name]["curations_path"] = None
                parser = instantiate(cfg["ontologies"]["parsers"][parser_name], _convert_="all")
                for generated_curated_term in parser.populate_databases(
                    return_curations=True, force=True
                ):
                    if generated_curated_term.source_term is None:
                        new_source_term_curations.add(generated_curated_term)
                    else:
                        new_generated_curations.add(generated_curated_term)

            original_terms_report = _OntologyUpgradeReport(
                existing_curations=existing_source_term_curations,
                curations_from_new_ontology_version=new_source_term_curations,
            )
            original_terms_report.match_existing_terms_to_new_terms()
            original_terms_report.write_curation_report(
                output_path=model_pack_path,
                parser_name=parser_name,
                prefix=_SOURCE_TERMS_PREFIX,
                curation_file_name=curations_path.name,
            )

            generated_terms_report = _OntologyUpgradeReport(
                existing_curations=existing_generated_curations,
                curations_from_new_ontology_version=new_generated_curations,
            )

            generated_terms_report.match_existing_terms_to_new_terms()

            generated_terms_report.write_curation_report(
                output_path=model_pack_path,
                parser_name=parser_name,
                prefix=_GENERATED_TERMS_PREFIX,
                curation_file_name=curations_path.name,
            )
