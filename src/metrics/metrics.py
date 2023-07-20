from sklearn import metrics


def precision(tp: int, fp: int):
    return tp / (tp + fp)


def npv(tn: int, fn: int):
    return tn / (tn + fn)


def recall(tp: int, fn: int):
    return tp / (tp + fn)


def sensitivity(tp: int, fn: int):
    """Alias for recall."""
    return recall(tp, fn)


def specificity(tn: int, fp: int):
    return tn / (tn + fp)


def accuracy(tp: int, tn: int, fp: int, fn: int):
    return (tp + tn) / (tp + tn + fp + fn)


def f1_score(tp: int, fp: int, fn: int):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 2 * prec * rec / (prec + rec)


def precision_score(y_true, y_pred) -> float:
    return metrics.precision_score(y_true, y_pred)


def recall_score(y_true, y_pred) -> float:
    return metrics.recall_score(y_true, y_pred)


def precision_recall_curve(y_true, probas_pred, pos_label=None):
    """
    Calculates the precision-recall curve.
    :param y_true: True binary labels.
        If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given
    :param probas_pred: Target scores, can either be probability estimates of the positive class,
        or non-thresholded measure of decisions
    :param pos_label: the label of the positive class
    :return: precision, recall, thresholds
    """
    return metrics.precision_recall_curve(y_true, probas_pred, pos_label=pos_label)
