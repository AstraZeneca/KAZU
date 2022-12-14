from typing import List

from seqeval.metrics import f1_score

IGNORE_IDX = -100


def accuracy(preds, labels):
    return (preds == labels).mean()


def numeric_label_f1_score(preds: List[List[int]], golds: List[List[int]], label_list) -> float:
    """
    Function to calculate F1 score using seqeval and numerical format labels.

    :param preds: 2d array of predicted label ids
    :param golds: 2d array of gold standard ids
    :param label_list: list of strings, for mappingids to labels
    :return:
    """

    pred_clean_labels_list = []
    gold_clean_labels_list = []

    assert len(preds) == len(golds)
    for preds_id_sequence, golds_id_sequence in zip(preds, golds):
        assert len(preds_id_sequence) == len(golds_id_sequence)
        p_labels = []
        g_labels = []
        for pred_label, gold_label in zip(preds_id_sequence, golds_id_sequence):
            if gold_label != IGNORE_IDX:
                p_labels.append(label_list[pred_label])
                g_labels.append(label_list[gold_label])
        pred_clean_labels_list.append(p_labels)
        gold_clean_labels_list.append(g_labels)

    return f1_score(gold_clean_labels_list, pred_clean_labels_list)
