import numpy as np
import torch


def to_numpy(inputs):
    if isinstance(inputs, list):
        return [to_numpy(x) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_numpy(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.cpu().detach().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError('only accept "tensor", "list", "dict", "numpy"')


def detach(inputs):
    if isinstance(inputs, list):
        return [detach(x) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: detach(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.cpu().detach()
    else:
        raise TypeError('only accept "tensor", "list", "dict"')


def default_collate(batches):
    fst_item = batches[0]
    if isinstance(fst_item, torch.Tensor):
        if len(fst_item.shape) == 0:
            return torch.stack(batches)
        else:
            return torch.cat(batches, dim=0)
    elif isinstance(fst_item, list):
        return [default_collate(tensors) for tensors in zip(*batches)]
    elif isinstance(fst_item, dict):
        return {key: default_collate([batch[key] for batch in batches]) for key in batches[0]}
    raise TypeError('only accept "tensor", "list", "dict"')
