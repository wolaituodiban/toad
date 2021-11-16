import torch
import toad


def test_metric():
    from sklearn.metrics import roc_auc_score, mean_squared_error
    history = toad.nn.History()
    history.log('pred', {'a': torch.tensor([0.2, 0.3])})
    history.log('label', {'a': torch.tensor([1, 0])})
    history.collate()
    metric = toad.utils.metric(roc_auc_score, label_first=True)
    metric = toad.nn.metric(metric)
    out = metric(history=history)
    assert out == {'a': 0.0}, out

    history = toad.nn.History()
    history.log('pred', {'a': {'c': torch.tensor([0.2, 0.3])}, 'b': [torch.tensor([0.2, 0.3])]})
    history.log('label', {'a': {'c': torch.tensor([1, 0])}, 'b': [torch.tensor([0.2, 0.3])]})
    history.collate()
    metric = toad.utils.metric([('$.a', roc_auc_score, True), ('$.b', mean_squared_error, True)])
    metric = toad.nn.metric(metric)
    out = metric(history=history)
    assert out == [{'c': 0.0}, [0.0]], out
