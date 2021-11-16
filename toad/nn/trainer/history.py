import torch
from .utils import default_collate, detach


class History:
    """model history
    """
    def __init__(self):
        self._store = {}

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        return self.log(key, value)

    def log(self, key, value):
        """log message to history

        Args:
            key (str): name of message
            value (any): tensor, list of tensors, dict of tensors
        """
        if key not in self._store:
            self._store[key] = []

        self._store[key].append(detach(value))

    def collate(self):
        self._store = {k: default_collate(v) for k, v in self._store.items()}
