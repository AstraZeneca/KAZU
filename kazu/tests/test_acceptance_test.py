import dataclasses
import itertools
from collections import defaultdict, Counter
from typing import List, Iterable, Dict, Set, Tuple, DefaultDict, Any

from kazu.data.data import Entity, Document
from kazu.pipeline import Pipeline, load_steps
from kazu.tests.utils import requires_model_pack, requires_label_studio
from kazu.utils.grouping import sort_then_group

pytestmark = [requires_model_pack, requires_label_studio]


NER_THRESHOLDS = {
    "gene": {"precision": 0.80, "recall": 0.80},
    "disease": {"precision": 0.70, "recall": 0.80},
    "drug": {"precision": 0.80, "recall": 0.80},
}

LINKING_THRESHOLDS = {
    "MONDO": {"precision": 0.70, "recall": 0.70},
    "MEDDRA": {"precision": 0.70, "recall": 0.70},
    "CHEMBL": {"precision": 0.80, "recall": 0.80},
    "ENSEMBL": {"precision": 0.80, "recall": 0.80},
}


def test_full_pipeline(capsys, override_kazu_test_config, label_studio_manager):

    cfg = override_kazu_test_config(
        # we don't want to test xref mappings, as we have no control over quality
        overrides=["pipeline=acceptance_test", "MappingStep=no_xref"],
    )
    pipeline = Pipeline(load_steps(cfg=cfg))
    analyse_full_pipeline(capsys, pipeline, label_studio_manager.export_from_ls())


def test_gold_standard_consistency(capsys, label_studio_manager):
    """
    a test that always passes, but reports potential inconsistencies in the gold standard

    :param label_studio_manager:
    :return:
    """
    check_annotation_consistency(capsys, label_studio_manager.export_from_ls())


class SectionScorer:
    def __init__(
        self,
        task: str,
        gold_ents: List[Entity],
        test_ents: List[Entity],
    ):
        """
        :param gold_ents:
        :param test_ents:
        """
        self.task = task
        self.gold_ents = gold_ents
        self.test_ents = test_ents
        self.gold_to_test_ent_soft: Dict[Entity, Set[Entity]] = defaultdict(set)

        self.ner_fp_soft: Set[Entity] = set(test_ents)
        self.ner_fn_soft: Set[Entity] = set(gold_ents)
        self.calculate_ner_matches()

        self.gold_to_test_mappings: Dict[
            Tuple[Entity, str], Dict[str, Tuple[str, str]]
        ] = defaultdict(dict)
        self.calculate_linking_matches()

    @staticmethod
    def group_mappings_by_source(ents: Iterable[Entity]) -> Dict[str, Set[Tuple[str, str]]]:
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
    docs: List[Document],
) -> Dict[str, List[SectionScorer]]:
    """
    score a list of documents by Section

    :param docs:
    :return: dict of entity class to one scorer per section
    """

    result = defaultdict(list)
    for doc in docs:
        for section in doc.sections:
            gold_ents_by_class = {
                k: list(v)
                for k, v in sort_then_group(
                    section.metadata["gold_entities"], key_func=lambda x: x.entity_class
                )
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
    fp_items_to_tasks: Dict[Any, set[str]] = dataclasses.field(default_factory=dict)
    fn_items_to_tasks: Dict[Any, set[str]] = dataclasses.field(default_factory=dict)

    def add_fp(self, item: Any, task: str) -> None:
        self.fp += 1
        self.fp_counter[item] += 1
        self.fp_items_to_tasks.setdefault(item, set()).add(task)

    def add_fn(self, item: Any, task: str) -> None:
        self.fn += 1
        self.fn_counter[item] += 1
        self.fn_items_to_tasks.setdefault(item, set()).add(task)

    def tasks_for_fp(self, items: List[Any]) -> Iterable[str]:
        return (
            task
            for item, tasks in self.fp_items_to_tasks.items()
            for task in tasks
            if item in items
        )

    def tasks_for_fn(self, items: List[Any]) -> Iterable[str]:
        return (
            task
            for item, tasks in self.fn_items_to_tasks.items()
            for task in tasks
            if item in items
        )

    @property
    def precision(self):
        div = self.fp + self.tp
        if div == 0:
            return 0.0
        else:
            return float(self.tp / div)

    @property
    def recall(self):
        div = self.fn + self.tp
        if div == 0:
            return 0.0
        else:
            return float(self.tp / div)

    @property
    def fp_info(self):
        return self.fp_counter.most_common()

    @property
    def fn_info(self):
        return self.fn_counter.most_common()


def aggregate_ner_results(
    class_and_scorers: Dict[str, List[SectionScorer]]
) -> Dict[str, AggregatedAccuracyResult]:
    result: Dict[str, AggregatedAccuracyResult] = {}
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
    class_and_scorers: Dict[str, List[SectionScorer]]
) -> Dict[str, AggregatedAccuracyResult]:

    result: DefaultDict[str, AggregatedAccuracyResult] = defaultdict(AggregatedAccuracyResult)
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
    capsys,
    results: Dict[str, AggregatedAccuracyResult],
    thresholds: Dict[str, Dict[str, float]],
):

    for key, threshold in thresholds.items():
        aggregated_result = results[key]
        prec = aggregated_result.precision
        rec = aggregated_result.recall
        message = "\n".join(f"{k} <incorrect {v} times>" for k, v in aggregated_result.fp_info)
        prec_message = (
            f"{aggregated_result.tp} / {aggregated_result.fp + aggregated_result.tp} \n{message}"
        )
        assert (
            prec >= threshold["precision"]
        ), f"{key} failed to meet precision threshold: {prec}. {prec_message}"
        with capsys.disabled():
            print(f"{key} passed precision threshold: {prec}. {prec_message}\n\n")

        message = "\n".join(f"{k} <missed {v} times>" for k, v in aggregated_result.fn_info)
        rec_message = (
            f"{aggregated_result.tp} / {aggregated_result.fn + aggregated_result.tp} \n{message}"
        )
        assert (
            rec >= threshold["recall"]
        ), f"{key} failed to meet recall threshold: {rec}. {rec_message}"
        with capsys.disabled():
            print(f"{key} passed recall threshold: {rec}. {rec_message}\n\n")


def analyse_full_pipeline(capsys, pipeline: Pipeline, docs: List[Document]):
    pipeline(docs)
    ner_dict = score_sections(docs)
    ner_results = aggregate_ner_results(ner_dict)
    check_results_meet_threshold(capsys, results=ner_results, thresholds=NER_THRESHOLDS)

    linking_results = aggregate_linking_results(ner_dict)
    check_results_meet_threshold(capsys, results=linking_results, thresholds=LINKING_THRESHOLDS)


def check_annotation_consistency(capsys, docs: List[Document]):
    all_ents = []
    ent_to_task_lookup: Dict[Entity, int] = {}  # used for reporting task id that may have issues
    for doc in docs:
        for section in doc.sections:
            ents = section.metadata["gold_entities"]
            all_ents.extend(ents)
            ent_to_task_lookup.update(
                {ent: str(section.metadata["label_studio_task_id"]) for ent in ents}
            )

    messages: DefaultDict[int, Set[str]] = defaultdict(set)
    # update the messages dict with any apparent issues
    for match_str, ents_iter in sort_then_group(all_ents, lambda x: x.match):
        ents = list(ents_iter)
        check_ent_match_abnormalities(ent_to_task_lookup, ents, match_str, messages)
        check_ent_class_consistency(ent_to_task_lookup, ents, match_str, messages)
        check_ent_mapping_consistency(ent_to_task_lookup, ents, match_str, messages)

    with capsys.disabled():
        print(f"{len(messages)} tasks with issues:\n")
        for doc_id in sorted(messages):
            print(f"\ntask id: {doc_id}\n" + "*" * 20 + "\n" + "\n".join(messages[doc_id]))


def check_ent_match_abnormalities(
    ent_to_task_lookup: Dict[Entity, int],
    ents: List[Entity],
    match_str: str,
    messages: Dict[int, Set[str]],
):
    """
    checks to see if any gold standard spans look a bit weird

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
    ent_to_task_lookup: Dict[Entity, int],
    ents: List[Entity],
    match_str: str,
    messages: Dict[int, Set[str]],
):
    """
    checks to see if any match strings have different entity_class information

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
    ent_to_task_lookup: Dict[Entity, int],
    ents: List[Entity],
    match_str: str,
    messages: Dict[int, Set[str]],
):
    """
    checks to see if any entity string matches have inconsistent mapping information

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
