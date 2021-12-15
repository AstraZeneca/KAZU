from typing import List

from seqeval.metrics import f1_score


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

    pred_labels_list = []
    labels_list = []
    for id_sequence in preds:
        pred_labels_list.append([label_list[ele] for ele in id_sequence])
    for id_sequence in golds:
        labels_list.append([label_list[ele] for ele in id_sequence])

    return f1_score(labels_list, pred_labels_list)
