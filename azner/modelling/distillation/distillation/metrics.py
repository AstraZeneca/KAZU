from seqeval.metrics import f1_score


def accuracy(preds, labels):
    return (preds == labels).mean()


def numeric_label_f1_score(preds, label_ids, label_list):
    """
    Function to calculate F1 score using seqeval and numerical format labels.
    preds, label_ids: list of list. Elements of the inner list are int/float type variables.
    """
    pred_labels_list = []
    labels_list = []
    for id_sequence in preds:
        pred_labels_list.append([label_list[ele] for ele in id_sequence])
    for id_sequence in label_ids.astype(int):
        labels_list.append([label_list[ele] for ele in id_sequence])

    return f1_score(labels_list, pred_labels_list)
