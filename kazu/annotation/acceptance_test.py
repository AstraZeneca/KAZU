import dataclasses
import itertools
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import cast, Any
from collections.abc import Iterable

import hydra
from hydra.utils import instantiate

from kazu.data import Entity, Document, IdsAndSource
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.utils.grouping import sort_then_group


class AcceptanceTestFailure(Exception):
    pass


AcceptanceCriteria = dict[str, dict[str, dict[str, float]]]


def acceptance_criteria() -> AcceptanceCriteria:
    with open(Path(os.environ["KAZU_MODEL_PACK"]).joinpath("acceptance_criteria.json")) as f:
        data = cast(AcceptanceCriteria, json.load(f))
    return data


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path=".", config_name="config")
def execute_full_pipeline_acceptance_test(cfg):
    manager = instantiate(cfg.LabelStudioManager)
    pipeline: Pipeline = instantiate(cfg.Pipeline)
    analyse_full_pipeline(pipeline, manager.export_from_ls(), acceptance_criteria())


class SectionScorer:
    def __init__(
        self,
        task: str,
        gold_ents: list[Entity],
        test_ents: list[Entity],
    ):
        """
        :param task:
        :param gold_ents:
        :param test_ents:
        """
        self.task = task
        self.gold_ents = gold_ents
        self.test_ents = test_ents
        self.gold_to_test_ent_soft: dict[Entity, set[Entity]] = defaultdict(set)

        self.ner_fp_soft: set[Entity] = set(test_ents)
        self.ner_fn_soft: set[Entity] = set(gold_ents)
        self.calculate_ner_matches()

        self.gold_to_test_mappings: dict[tuple[Entity, str], dict[str, IdsAndSource]] = defaultdict(
            dict
        )
        self.calculate_linking_matches()

    @staticmethod
    def group_mappings_by_source(ents: Iterable[Entity]) -> dict[str, IdsAndSource]:
        mappings_by_source = defaultdict(set)
        for ent in ents:
            for mapping in ent.mappings:
                mappings_by_source[mapping.source].add(
                    (
                        mapping.source,
                        f"{mapping.default_label}|{mapping.idx}",
                    )
                )
        return dict(mappings_by_source)

    def calculate_ner_matches(self):
        combos = itertools.product(self.gold_ents, self.test_ents)
        for (gold_ent, test_ent) in combos:
            if (
                gold_ent.spans == test_ent.spans or gold_ent.is_partially_overlapped(test_ent)
            ) and gold_ent.entity_class == test_ent.entity_class:
                self.gold_to_test_ent_soft[gold_ent].add(test_ent)
                self.ner_fp_soft.discard(test_ent)
                self.ner_fn_soft.discard(gold_ent)

    def calculate_linking_matches(self):
        for gold_ent, test_ents in self.gold_to_test_ent_soft.items():
            gold_mappings_by_source = self.group_mappings_by_source([gold_ent])
            test_mappings_by_source = self.group_mappings_by_source(test_ents)
            sources = set(gold_mappings_by_source.keys()).union(test_mappings_by_source.keys())
            for source in sources:
                gold_mappings = gold_mappings_by_source.get(source, set())
                test_mappings_set = test_mappings_by_source.get(source, set())
                tp = gold_mappings.intersection(test_mappings_set)
                fn = gold_mappings - test_mappings_set
                fp = test_mappings_set - gold_mappings
                self.gold_to_test_mappings[
                    (
                        gold_ent,
                        source,
                    )
                ] = {"tp": tp, "fp": fp, "fn": fn}


def score_sections(
    docs: list[Document],
) -> dict[str, list[SectionScorer]]:
    """Score a list of documents by Section.

    :param docs:
    :return: dict of entity class to one scorer per section
    """

    result = defaultdict(list)
    for doc in docs:
        for section in doc.sections:
            gold_ents: list[Entity] = section.metadata["gold_entities"]
            gold_ents_by_class = {
                k: list(v) for k, v in sort_then_group(gold_ents, key_func=lambda x: x.entity_class)
            }
            for entity_class, test_ents in sort_then_group(
                section.entities, key_func=lambda x: x.entity_class
            ):
                scorer = SectionScorer(
                    task=section.metadata["label_studio_task_id"],
                    gold_ents=gold_ents_by_class.get(entity_class, []),
                    test_ents=list(test_ents),
                )
                result[entity_class].append(scorer)

    return dict(result)


@dataclasses.dataclass
class AggregatedAccuracyResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fp_counter: Counter = dataclasses.field(default_factory=Counter)
    fn_counter: Counter = dataclasses.field(default_factory=Counter)
    fp_items_to_tasks: dict[Any, set[str]] = dataclasses.field(default_factory=dict)
    fn_items_to_tasks: dict[Any, set[str]] = dataclasses.field(default_factory=dict)

    def add_fp(self, item: Any, task: str) -> None:
        self.fp += 1
        self.fp_counter[item] += 1
        self.fp_items_to_tasks.setdefault(item, set()).add(task)

    def add_fn(self, item: Any, task: str) -> None:
        self.fn += 1
        self.fn_counter[item] += 1
        self.fn_items_to_tasks.setdefault(item, set()).add(task)

    def tasks_for_fp(self, items: list[Any]) -> Iterable[str]:
        return (
            task
            for item, tasks in self.fp_items_to_tasks.items()
            for task in tasks
            if item in items
        )

    def tasks_for_fn(self, items: list[Any]) -> Iterable[str]:
        return (
            task
            for item, tasks in self.fn_items_to_tasks.items()
            for task in tasks
            if item in items
        )

    @property
    def precision(self) -> float:
        div = self.fp + self.tp
        if div == 0:
            return 0.0
        else:
            return float(self.tp / div)

    @property
    def recall(self) -> float:
        div = self.fn + self.tp
        if div == 0:
            return 0.0
        else:
            return float(self.tp / div)

    @property
    def fp_info(self) -> list[Any]:
        return self.fp_counter.most_common()

    @property
    def fn_info(self) -> list[Any]:
        return self.fn_counter.most_common()


def aggregate_ner_results(
    class_and_scorers: dict[str, list[SectionScorer]]
) -> dict[str, AggregatedAccuracyResult]:
    result: dict[str, AggregatedAccuracyResult] = {}
    for ent_class, scorers in class_and_scorers.items():
        acc_result = AggregatedAccuracyResult()
        for scorer in scorers:
            acc_result.tp += len(scorer.gold_to_test_ent_soft)
            for ent in scorer.ner_fn_soft:
                acc_result.add_fn(item=ent.match, task=scorer.task)
            for ent in scorer.ner_fp_soft:
                acc_result.add_fp(item=ent.match, task=scorer.task)
        result[ent_class] = acc_result
    return result


def aggregate_linking_results(
    class_and_scorers: dict[str, list[SectionScorer]]
) -> dict[str, AggregatedAccuracyResult]:

    result: defaultdict[str, AggregatedAccuracyResult] = defaultdict(AggregatedAccuracyResult)
    for _, scorers in class_and_scorers.items():
        for scorer in scorers:
            for (
                _,
                source,
            ), mapping_result in scorer.gold_to_test_mappings.items():
                acc_result = result[source]
                acc_result.tp += len(mapping_result["tp"])
                for fn_item in mapping_result["fn"]:
                    acc_result.add_fn(item=fn_item, task=scorer.task)
                for fp_item in mapping_result["fp"]:
                    acc_result.add_fp(item=fp_item, task=scorer.task)
    return dict(result)


def check_results_meet_threshold(
    results: dict[str, AggregatedAccuracyResult],
    thresholds: dict[str, dict[str, float]],
) -> None:

    for key, threshold in thresholds.items():
        aggregated_result = results[key]
        prec = aggregated_result.precision
        rec = aggregated_result.recall
        message = "\n".join(f"{k} <incorrect {v} times>" for k, v in aggregated_result.fp_info)
        prec_message = (
            f"{aggregated_result.tp} / {aggregated_result.fp + aggregated_result.tp} \n{message}"
        )
        if prec >= threshold["precision"]:
            print(f"{key} passed precision threshold: {prec}. {prec_message}\n\n")
        else:
            raise AcceptanceTestFailure(
                f"{key} failed to meet precision threshold: {prec}. {prec_message}"
            )

        message = "\n".join(f"{k} <missed {v} times>" for k, v in aggregated_result.fn_info)
        rec_message = (
            f"{aggregated_result.tp} / {aggregated_result.fn + aggregated_result.tp} \n{message}"
        )
        if rec >= threshold["recall"]:
            print(f"{key} passed recall threshold: {rec}. {rec_message}\n\n")
        else:
            raise AcceptanceTestFailure(
                f"{key} failed to meet recall threshold: {rec}. {rec_message}"
            )


def analyse_full_pipeline(
    pipeline: Pipeline,
    docs: list[Document],
    acceptance_criteria: dict[str, dict[str, dict[str, float]]],
) -> None:
    pipeline(docs)
    ner_dict = score_sections(docs)
    ner_results = aggregate_ner_results(ner_dict)
    check_results_meet_threshold(
        results=ner_results, thresholds=acceptance_criteria["NER_THRESHOLDS"]
    )

    linking_results = aggregate_linking_results(ner_dict)
    check_results_meet_threshold(
        results=linking_results, thresholds=acceptance_criteria["LINKING_THRESHOLDS"]
    )


def analyse_annotation_consistency(docs: list[Document]) -> None:
    all_ents: list[Entity] = []
    ent_to_task_lookup: dict[Entity, str] = {}  # used for reporting task id that may have issues
    for doc in docs:
        for section in doc.sections:
            ents: list[Entity] = section.metadata["gold_entities"]
            all_ents.extend(ents)
            ent_to_task_lookup.update(
                {ent: str(section.metadata["label_studio_task_id"]) for ent in ents}
            )

    messages: defaultdict[str, set[str]] = defaultdict(set)
    # update the messages dict with any apparent issues
    for match_str, ents_iter in sort_then_group(all_ents, lambda x: x.match):
        ents = list(ents_iter)
        check_ent_match_abnormalities(ent_to_task_lookup, ents, match_str, messages)
        check_ent_class_consistency(ent_to_task_lookup, ents, match_str, messages)
        check_ent_mapping_consistency(ent_to_task_lookup, ents, match_str, messages)

    print(f"{len(messages)} tasks with issues:\n")
    for doc_id in sorted(messages):
        print(f"\ntask id: {doc_id}\n" + "*" * 20 + "\n" + "\n".join(messages[doc_id]))


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../../", config_name="conf")
def check_annotation_consistency(cfg):

    manager = instantiate(cfg.LabelStudioManager)
    docs = manager.export_from_ls()
    analyse_annotation_consistency(docs)


def check_ent_match_abnormalities(
    ent_to_task_lookup: dict[Entity, str],
    ents: list[Entity],
    match_str: str,
    messages: dict[str, set[str]],
) -> None:
    """Checks to see if any gold standard spans look a bit weird.

    :param ent_to_task_lookup:
    :param ents:
    :param match_str:
    :param messages:
    :return:
    """
    if len(match_str) == 1 or (
        len(match_str) == 2 and not all(char.isalnum() for char in match_str)
    ):
        for ent in ents:
            messages[ent_to_task_lookup[ent]].add(
                f"WARNING: ent string <{match_str}> may be abnormal"
            )


def check_ent_class_consistency(
    ent_to_task_lookup: dict[Entity, str],
    ents: list[Entity],
    match_str: str,
    messages: dict[str, set[str]],
) -> None:
    """Checks to see if any match strings have different entity_class information.

    :param ent_to_task_lookup:
    :param ents:
    :param match_str:
    :param messages:
    :return:
    """
    by_doc = defaultdict(set)
    all_entity_classes = set()
    for ent in ents:
        by_doc[ent_to_task_lookup[ent]].add(ent.entity_class)
        all_entity_classes.add(ent.entity_class)

    if len(all_entity_classes) > 1:
        message = "\n".join(f"{doc_id}:{classes}" for doc_id, classes in by_doc.items())
        for doc_id, classes in by_doc.items():
            messages[doc_id].add(
                f"WARNING: ent string <{match_str}> is class confused in the following documents:\n{message}"
            )


def check_ent_mapping_consistency(
    ent_to_task_lookup: dict[Entity, str],
    ents: list[Entity],
    match_str: str,
    messages: dict[str, set[str]],
) -> None:
    """Checks to see if any entity string matches have inconsistent mapping information.

    :param ent_to_task_lookup:
    :param ents:
    :param match_str:
    :param messages:
    :return:
    """
    mappings_to_doc_id = defaultdict(set)
    for ent in ents:
        mappings_to_doc_id[frozenset(ent.mappings)].add(ent_to_task_lookup[ent])
    if len(mappings_to_doc_id) > 1:
        group_definitions = set()
        doc_id_to_groups = defaultdict(set)
        for i, (mappings, doc_ids) in enumerate(mappings_to_doc_id.items()):
            group_key = f"group_{i}:\n"
            group_msg = (
                "\n".join(
                    f"{mapping.source}|{mapping.default_label}|{mapping.idx}"
                    for mapping in mappings
                )
                if len(mappings) > 0
                else "<NONE>"
            )
            group_msg = group_key + group_msg + "\n"
            group_definitions.add(group_msg)
            for doc_id in doc_ids:
                doc_id_to_groups[doc_id].add(group_key)

        group_definition_message = "\n".join(group_definitions)
        confused_docs_message = "\n".join(
            f"{doc_id}:{sorted(groups)}" for doc_id, groups in sorted(doc_id_to_groups.items())
        )

        overall_message = (
            f"WARNING: ent string <{match_str}> has inconsistent mappings. This may be a genuine ambiguity or "
            f"a mistake in annotation: \n\n {group_definition_message} \n\naffected tasks:"
            f"\n {confused_docs_message}\n"
        )
        for doc_id in doc_id_to_groups:
            messages[doc_id].add(overall_message)


if __name__ == "__main__":
    execute_full_pipeline_acceptance_test()
