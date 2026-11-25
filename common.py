import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

SEED = 260003
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SPLITS = ROOT / 'splits'
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
FIGURES = ROOT / 'figures'
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def setup(seed=SEED):
    for directory in (DATA, SPLITS, RESULTS, CHECKPOINTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def datasets_from_saved_split():
    split = SPLITS / 'cifar10_seed_260003.npz'
    if not split.exists():
        raise FileNotFoundError(f'Run python prepare_data.py first: {split}')
    indices = np.load(split)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_train = CIFAR10(DATA, train=True, download=False, transform=transform)
    test = CIFAR10(DATA, train=False, download=False, transform=transform)
    train_idx, val_idx = indices['train_idx'], indices['val_idx']
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    return Subset(full_train, train_idx.tolist()), Subset(full_train, val_idx.tolist()), test


def loader(dataset, batch_size=128, shuffle=False, workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


class MLP(nn.Module):
    def __init__(self, hidden=(512, 256), batchnorm=False):
        super().__init__()
        dims = (3072, *hidden, 10)
        layers = [nn.Flatten()]
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return {'loss': loss_sum / total, 'accuracy': 100 * correct / total}


def fit(model, train, val, name, epochs, device, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_accuracy, best_epoch = [], -1, None
    for e in range(1, epochs+1):
        start = time.perf_counter()
        tr = epoch(model, train, device, optimizer)
        va = epoch(model, val, device)
        item = {'epoch': e, 'train_loss': tr['loss'], 'train_accuracy': tr['accuracy'],
                'val_loss': va['loss'], 'val_accuracy': va['accuracy'],
                'seconds': time.perf_counter()-start}
        history.append(item)
        if va['accuracy'] > best_accuracy:
            best_accuracy, best_epoch = va['accuracy'], e
            torch.save({'state_dict': model.state_dict(), 'epoch': e,
                        'val_accuracy': best_accuracy}, CHECKPOINTS / f'{name}.pt')
        print(f"{name} epoch={e:03d} train={tr['accuracy']:.2f}% "
              f"val={va['accuracy']:.2f}% loss={va['loss']:.4f} "
              f"seconds={item['seconds']:.1f}", flush=True)
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(RESULTS / f'{name}_history.csv', index=False)
    summary = {'model': name, 'parameters': parameter_count(model), 'best_val_accuracy':
               best_accuracy, 'best_epoch': best_epoch, 'total_seconds': df.seconds.sum(),
               'epochs': epochs, 'lr': lr, 'weight_decay': weight_decay}
    (RESULTS / f'{name}_summary.json').write_text(json.dumps(summary, indent=2))
    return df, summary


def load_best(model, name, device):
    checkpoint = torch.load(CHECKPOINTS / f'{name}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)


def plot_histories(named_histories, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, hist in named_histories.items():
        axes[0].plot(hist.epoch, hist.train_loss, label=f'{name} train')
        axes[0].plot(hist.epoch, hist.val_loss, '--', label=f'{name} val')
        axes[1].plot(hist.epoch, hist.train_accuracy, label=f'{name} train')
        axes[1].plot(hist.epoch, hist.val_accuracy, '--', label=f'{name} val')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


class NoisyLabels(Dataset):
    def __init__(self, train_subset, fraction=0.20, seed=SEED):
        self.dataset = train_subset
        all_targets = np.asarray(train_subset.dataset.targets)
        self.clean = all_targets[np.asarray(train_subset.indices)].copy()
        self.labels = self.clean.copy()
        rng = np.random.default_rng(seed)
        self.corrupted = np.zeros(len(self.labels), dtype=bool)
        chosen = rng.choice(len(self.labels), int(len(self.labels)*fraction), replace=False)
        self.corrupted[chosen] = True
        self.labels[chosen] = (self.labels[chosen] + rng.integers(1, 10, len(chosen))) % 10

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, int(self.labels[index])


@torch.no_grad()
def memorization_scores(model, dataset, device, batch_size=128, workers=2):
    model.eval()
    predictions = []
    for x, _ in loader(dataset, batch_size, False, workers):
        predictions.append(model(x.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(predictions)
    good = ~dataset.corrupted
    noisy = dataset.corrupted
    return {
        'clean_train_accuracy': 100 * np.mean(pred[good] == dataset.clean[good]),
        'corrupted_label_fit': 100 * np.mean(pred[noisy] == dataset.labels[noisy]),
        'corrupted_subset_true_accuracy': 100 * np.mean(pred[noisy] == dataset.clean[noisy])
    }
