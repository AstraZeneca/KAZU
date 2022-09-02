import itertools
from collections import defaultdict, Counter
from typing import List, Iterable, Dict, Set, Tuple, DefaultDict

import pytest
from kazu.data.data import Entity, Document
from kazu.pipeline import Pipeline, load_steps
from kazu.tests.utils import requires_model_pack, requires_gold_standard

# this applies the require_model_pack mark to all tests in this module
from kazu.utils.grouping import sort_then_group

pytestmark = requires_model_pack


NER_THRESHOLDS = {
    "gene": {"precision": 0.30, "recall": 0.30},
    "disease": {"precision": 0.30, "recall": 0.30},
    # "drug": {"precision": 0.30, "recall": 0.30},
}

LINKING_THRESHOLDS = {
    "MONDO": {"precision": 0.30, "recall": 0.30},
    "MEDDRA": {"precision": 0.30, "recall": 0.30},
    # "CHEMBL": {"precision": 0.30, "recall": 0.30},
    "ENSEMBL": {"precision": 0.30, "recall": 0.30},
}


def precision(tp: int, fp: int) -> float:
    div = float(fp) + float(tp)
    if div == 0.0:
        return 0.0
    else:
        return float(tp) / div


def recall(tp: int, fn: int) -> float:
    div = float(fn) + float(tp)
    if div == 0.0:
        return 0.0
    else:
        return float(tp) / div


class NerScoring:
    def __init__(
        self,
        gold_ents: List[Entity],
        test_ents: List[Entity],
    ):
        """
        :param gold_ents:
        :param test_ents:
        """
        self.test_ents = test_ents
        self.gold_ents = gold_ents
        self.gold_to_test_ent_hard_mapping: Dict[Entity, Set[Entity]] = defaultdict(set)
        self.gold_to_test_ent_soft_mapping: Dict[Entity, Set[Entity]] = defaultdict(set)

        self.ner_fp_soft: Set[Entity] = set(test_ents)
        self.ner_fp_hard: Set[Entity] = set(test_ents)
        self.ner_fn_soft: Set[Entity] = set(gold_ents)
        self.ner_fn_hard: Set[Entity] = set(gold_ents)
        self.calculate_ner_matches()

        self.gold_to_test_mapping_hard_mapping: Dict[
            Tuple[Entity, str], Dict[str, Tuple[str, str]]
        ] = defaultdict(dict)
        self.gold_to_test_mapping_soft_mapping: Dict[
            Tuple[Entity, str], Dict[str, Tuple[str, str]]
        ] = defaultdict(dict)
        self.calculate_linking_matches()

        # self.precision_hard = precision(
        #     tp=len(self.gold_to_test_ent_hard_mapping), fp=len(self.ner_fp_hard)
        # )
        # self.precision_soft = precision(
        #     tp=len(self.gold_to_test_ent_soft_mapping), fp=len(self.ner_fp_soft)
        # )
        # self.recall_hard = recall(tp=len(self.gold_to_test_ent_hard_mapping), fn=len(self.ner_fn_hard))
        # self.recall_soft = recall(tp=len(self.gold_to_test_ent_soft_mapping), fn=len(self.ner_fn_soft))

    def calculate_ner_matches(self):
        combos = itertools.product(self.gold_ents, self.test_ents)
        for combo in combos:
            gold_ent, test_ent = combo
            # exact matches
            if (
                gold_ent.start == test_ent.start
                and gold_ent.end == test_ent.end
                and gold_ent.entity_class == test_ent.entity_class
            ):
                self.gold_to_test_ent_hard_mapping[gold_ent].add(test_ent)
                self.ner_fp_hard.discard(test_ent)
                self.ner_fn_hard.discard(gold_ent)
            # partial matches (a superset of exact matches)
            if (
                gold_ent.is_partially_overlapped(test_ent)
                and gold_ent.entity_class == test_ent.entity_class
            ):
                self.gold_to_test_ent_soft_mapping[gold_ent].add(test_ent)
                self.ner_fp_soft.discard(test_ent)
                self.ner_fn_soft.discard(gold_ent)

    def calculate_linking_matches(self):
        for gold_ent, test_ents in self.gold_to_test_ent_soft_mapping.items():
            gold_mappings_by_source, test_mappings_by_source = group_mappings_by_source(
                gold_ent, test_ents
            )
            for source, gold_mappings in gold_mappings_by_source:
                gold_mappings_set = set(gold_mappings)
                test_mappings_set = test_mappings_by_source.get(source, set())
                tp = gold_mappings_set.intersection(test_mappings_set)
                fn = gold_mappings_set - test_mappings_set
                fp = test_mappings_set - gold_mappings_set
                self.gold_to_test_mapping_soft_mapping[
                    (
                        gold_ent,
                        source,
                    )
                ] = {"tp": tp, "fp": fp, "fn": fn}


def analyse_ner(
    docs: List[Document],
) -> Dict[str, List[NerScoring]]:

    result = defaultdict(list)
    for doc in docs:
        for section in doc.sections:
            for entity_class, ents in sort_then_group(
                section.entities, key_func=lambda x: x.entity_class
            ):
                gold_ents = section.metadata["gold_entities"]
                test_ents = section.entities
                scorer = NerScoring(gold_ents=gold_ents, test_ents=test_ents)
                result[entity_class].append(scorer)

    return dict(result)


def aggregate_ner_results(
    class_and_scorers: Dict[str, List[NerScoring]]
) -> Dict[str, Dict[str, int]]:
    result: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    for ent_class, scorers in class_and_scorers.items():
        tp_hard = 0
        fn_hard = 0
        fp_hard = 0
        tp_soft = 0
        fn_soft = 0
        fp_soft = 0
        for scorer in scorers:
            tp_hard += len(scorer.gold_to_test_ent_hard_mapping)
            tp_soft += len(scorer.gold_to_test_ent_soft_mapping)
            fn_hard += len(scorer.ner_fn_hard)
            fn_soft += len(scorer.ner_fn_soft)
            fp_hard += len(scorer.ner_fp_hard)
            fp_soft += len(scorer.ner_fp_soft)

        result[ent_class]["tp_hard"] = tp_hard
        result[ent_class]["tp_soft"] = tp_soft
        result[ent_class]["fp_hard"] = fp_hard
        result[ent_class]["fp_soft"] = fp_soft
        result[ent_class]["fn_hard"] = fn_hard
        result[ent_class]["fn_soft"] = fn_soft
    return dict(result)


def aggregate_linking_results(
    class_and_scorers: Dict[str, List[NerScoring]]
) -> Dict[str, Dict[str, int]]:
    result: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for _, scorers in class_and_scorers.items():
        for scorer in scorers:
            for (
                _,
                source,
            ), results in scorer.gold_to_test_mapping_soft_mapping.items():
                result[source]["tp"] = result[source]["tp"] + len(results["tp"])
                result[source]["fp"] = result[source]["fp"] + len(results["fp"])
                result[source]["fn"] = result[source]["fn"] + len(results["fn"])
    return dict(result)


def group_mappings_by_source(gold_ent: Entity, test_ents: Iterable[Entity]):
    all_gold_mappings: Set[Tuple[str, str]] = set(
        (
            x.source,
            x.idx,
        )
        for x in gold_ent.mappings
    )
    all_test_mappings: Set[Tuple[str, str]] = set(
        (
            mapping.source,
            mapping.idx,
        )
        for ent in test_ents
        for mapping in ent.mappings
    )
    gold_mappings_by_source = sort_then_group(all_gold_mappings, key_func=lambda x: x[0])
    test_mappings_by_source = {
        k: set(v) for k, v in sort_then_group(all_test_mappings, key_func=lambda x: x[0])
    }
    return gold_mappings_by_source, test_mappings_by_source


def check_ner_results_meet_threshold(
    results: Dict[str, Dict[str, int]],
    thresholds: Dict[str, Dict[str, float]],
    ner_scorer_dict: Dict[str, List[NerScoring]],
):

    for key, threshold in thresholds.items():
        tp_soft = results[key]["tp_soft"]
        fp_soft = results[key]["fp_soft"]
        fn_soft = results[key]["fn_soft"]
        prec = precision(tp=tp_soft, fp=fp_soft)
        rec = recall(tp=tp_soft, fn=fn_soft)
        if prec < threshold["precision"]:
            fp_ents = itertools.chain.from_iterable(
                scorer.ner_fp_soft for scorer in ner_scorer_dict[key]
            )
            fp_matches = Counter(ent.match for ent in fp_ents)
            message = "\n".join(
                f"{k} <missed {v} times>" for k, v in fp_matches.most_common(len(fp_matches))
            )

            pytest.fail(
                f"{key} failed to meet precision threshold: {prec}. {tp_soft} / {fp_soft +tp_soft} \n{message}"
            )
        if rec < threshold["recall"]:

            missed_ents = itertools.chain.from_iterable(
                scorer.ner_fn_soft for scorer in ner_scorer_dict[key]
            )
            missed_matches = Counter(ent.match for ent in missed_ents)
            message = "\n".join(
                f"{k} <missed {v} times>"
                for k, v in missed_matches.most_common(len(missed_matches))
            )
            pytest.fail(
                f"{key} failed to meet recall threshold: {rec}. {tp_soft} / {fn_soft +tp_soft} \n{message}"
            )


def check_linking_results_meet_threshold(results: Dict[str, Dict[str, int]]):

    for source, threshold in LINKING_THRESHOLDS.items():
        tp = results[source]["tp"]
        fp = results[source]["fp"]
        fn = results[source]["fn"]
        prec = precision(tp=tp, fp=fp)
        rec = recall(tp=tp, fn=fn)
        if prec < threshold["precision"] and fp > 0:
            pytest.fail(f"{source} failed to meet precision threshold: {prec}. {tp} / {fp +tp}")
        if rec < threshold["recall"] and fn > 0:
            pytest.fail(f"{source} failed to meet recall threshold: {rec}. {tp} / {fn +tp}")


def analyse_full_pipeline(pipeline: Pipeline, docs: List[Document]):
    pipeline(docs)
    ner_dict = analyse_ner(docs)
    ner_results = aggregate_ner_results(ner_dict)
    check_ner_results_meet_threshold(
        results=ner_results, thresholds=NER_THRESHOLDS, ner_scorer_dict=ner_dict
    )

    linking_results = aggregate_linking_results(ner_dict)
    check_linking_results_meet_threshold(linking_results)


@requires_gold_standard
@requires_model_pack
def test_full_pipeline(override_kazu_test_config, acceptance_test_docs):

    cfg = override_kazu_test_config(
        overrides=["pipeline=acceptance_test"],
    )

    # TODO - needs futher work/testing
    pipeline = Pipeline(load_steps(cfg=cfg))
    analyse_full_pipeline(pipeline, acceptance_test_docs)
