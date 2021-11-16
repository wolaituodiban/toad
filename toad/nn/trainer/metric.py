from .callback import callback
from .utils import to_numpy


class metric(callback):
    """
    given a json format input, several json path and metrics func
    return metric value for certern json path and metric
    Examples:
        >>> from sklearn.metrics import roc_auc_score
        ... history.log('pred', ...)
        ... history.log('label', ...)
        ... history.collate()
        ... auc = metric(roc_auc_score)
        ... auc(history)
        0.5

    Args:
        x: Callable or list of tuple(json path (str), function (callable), label_first (bool))
        label_fist: if is not None, func should be a single function

    Returns:
        new function
    """
    def wrapper(self, *args, **kwargs):
        history = kwargs['history']
        return self.call(to_numpy(history['pred']), to_numpy(history['label']))
