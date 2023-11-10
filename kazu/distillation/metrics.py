try:
    from seqeval.metrics import f1_score
except ImportError as e:
    raise ImportError(
        "Running the model distillation code requires seqeval to be installed.\n"
        "We recommend running 'pip install kazu[model_training]' to get all model training"
        " dependencies."
    ) from e


IGNORE_IDX = -100


def accuracy(preds, labels):
    return (preds == labels).mean()


def numeric_label_f1_score(
    preds: list[list[int]], golds: list[list[int]], label_list: list[str]
) -> float:
    """Function to calculate F1 score using seqeval and numerical format labels.

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
    f1: float = f1_score(gold_clean_labels_list, pred_clean_labels_list)
    return f1
