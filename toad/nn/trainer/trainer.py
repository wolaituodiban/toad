from typing import Dict, Callable

import torch
from torch import optim

from .callback import callback as Callback
from .earlystop import earlystopping
from .history import History
from .metric import metric
from ..module import Module

from ...utils.progress import Progress


class Trainer:
    """trainer for training models
    """
    def __init__(self, model, loader = None, optimizer = None, loss = None, keep_history = None,
                 early_stopping: earlystopping = None, metrics: Dict[str, Callable] = {}, log_forward=True):
        """initialization

        Args:
            model (nn.Module): model will be trained
            loader (torch.DataLoader): training data loader
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            loss (Callable): could be called as 'loss(y_hat, y)'
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            metrics (dict(str, callable)): name and metric function which will be called as metric_func(y_hat, y)
            keep_history (int): keep the last n-th epoch logs, `None` will keep all
            log_forward (bool): auto log output of forward
        """
        self.model = model
        self.loader = loader

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
        
        self.optimizer = optimizer

        self.loss = loss

        # set default early stopping
        if early_stopping is None:
            from .earlystop import loss_scoring
            early_stopping = loss_scoring
        
        self.early_stop = early_stopping
        self.metrics = [metric(name=k)(v) for k, v in metrics.items()]
        self.log_forward = log_forward

        from collections import deque
        self.history = deque(maxlen = keep_history)
        self._current_history: History = None

    def fit_step(self, batch):
        if isinstance(self.model, Module):
            l = self.model.fit_step(batch)
        else:
            pred = self.model(batch[0])
            l = self.loss(pred, batch[1])
            if self.log_forward:
                self._current_history.log('pred', pred)
        self._current_history.log('loss', l)
        self._current_history.log('label', batch[1])
        return l

    def train(self, loader = None, valid_loader = None, epoch = 10, callbacks = [], start = 0, backward_rounds = 1):
        """
        Args:
            loader (torch.DataLoader): training data loader
            epoch (int): number of epoch for training loop
            callbacks (callable): callable function will be called every epoch
                - parameters of callback
                    model (nn.Module): the training model
                    history (History): history of total log records
                    epoch (int): current epoch number
                    trainer (Trainer): self trainer
            start (int): epoch start from n round
            backward_rounds (int): backward after every n rounds 
        
        Returns:
            Module: the model with best performents
        """
        if loader is not None:
            self.loader = loader
        
        if self.loader is None:
            raise ValueError("loader is not set, please set a loader for trainning!")

        callbacks = [c if isinstance(c, Callback) else Callback(c) for c in callbacks]
        
        # init progress bar
        p = Progress(self.loader)

        for ep in range(start, epoch):
            # set model to train mode
            self.model.train()

            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            self._current_history = History()
            self.history.append(self._current_history)

            loss = 0.
            backward_loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                l = self.fit_step(batch)
                
                backward_loss = l + backward_loss
                if i % backward_rounds == 0 or i == len(p):
                    self.optimizer.zero_grad()
                    backward_loss.backward()
                    self.optimizer.step()
                    
                    # reset backward loss
                    backward_loss = 0.

                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)

            # collate current history
            self._current_history.collate()

            # setup callback params
            callback_params = {
                "model": self.model.eval(),
                "history": self._current_history,
                "epoch": ep,
                "trainer": self,
            }

            with torch.no_grad():
                for metric in self.metrics:
                    print('{}:'.format(metric.name), metric(**callback_params))

                if self.metrics and valid_loader:
                    self.evaluate(valid_loader, callbacks = self.metrics)

                for c in callbacks:
                    c(**callback_params)
                    
                if self.early_stop and self.early_stop(**callback_params):
                    # set best state to model
                    best_state = self.early_stop.get_best_state()
                    self.model.load_state_dict(best_state)
                    return self.model

        return self.model
    
    @torch.no_grad()
    def evaluate(self, loader, callbacks = []):
        """evalute model

        Args:
            loader (torch.DataLoader): evaluation data loader
            callback (callable): callback function
        """
        callbacks = [c if isinstance(c, Callback) else Callback(c) for c in callbacks]
        
        # init progress bar
        p = Progress(loader)
        p.prefix = f"Evaluate"

        self._current_history = History()
        self.model.eval()
        
        loss = 0.
        for i, batch in enumerate(p, start = 1):
            # step fit
            l = self.fit_step(batch)

            loss += (l.item() - loss) / i
            p.suffix = 'loss:{:.4f}'.format(loss)
        self._current_history.collate()

        callback_params = {
            "model": self.model.eval(),
            "history": self._current_history,
            "trainer": self,
            "epoch": None
        }

        for metric in self.metrics:
            print('{}:'.format(metric.name), metric(**callback_params))

        for c in callbacks:
            c(**callback_params)

        return self._current_history
