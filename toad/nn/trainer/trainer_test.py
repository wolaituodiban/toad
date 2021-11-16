import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from toad.nn import Module, Trainer, earlystopping
from toad.utils import metric


DATASET_SIZE = 20000
NUM_FEATS = 784
NUM_CLASSES = 2

X = torch.rand(DATASET_SIZE, NUM_FEATS, dtype = torch.float)
y = torch.randint(NUM_CLASSES, size = (DATASET_SIZE,), dtype = torch.long)

loader = DataLoader(
    TensorDataset(X, y),
    batch_size = 128,
    shuffle = True,
)

class TestModel(Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.linear = nn.Linear(in_feats, out_feats)
    
    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)
    
    def fit_step(self, batch):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)


def test_trainer():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model, loader)
    trainer.train(epoch = 2)
    assert len(trainer.history) == 2


def test_trainer_early_stopping():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    
    @earlystopping(delta = -1.0, patience = 3)
    def scoring(history):
        return history['loss'].mean()

    trainer = Trainer(model, loader, early_stopping = scoring)
    trainer.train(valid_loader=loader, epoch = 200)
    assert len(trainer.history) == 4


def test_train_torch_module():
    from sklearn.metrics import roc_auc_score, mean_squared_error

    class TestModel2(torch.nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()

            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, x):
            x = self.linear(x)
            return F.relu(x.view(x.shape[0]))

    X = torch.rand(DATASET_SIZE, NUM_FEATS, dtype=torch.float)
    y = torch.randint(2, size=(DATASET_SIZE, ), dtype=torch.float)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size = 128,
        shuffle = True,
    )

    model = TestModel2(NUM_FEATS, 1)

    metrics = {
        'auc': metric(roc_auc_score, label_first=True),
        'mse': metric(mean_squared_error, label_first=True)
    }
    trainer = Trainer(model, loss=F.binary_cross_entropy_with_logits, metrics=metrics, log_forward=True)
    trainer.train(loader, valid_loader=loader, epoch=2)
    assert len(trainer.history) == 2
