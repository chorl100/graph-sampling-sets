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
