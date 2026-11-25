import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from common import (SEED, MEAN, STD, RESULTS, CHECKPOINTS, FIGURES,
                    setup, datasets_from_saved_split, loader, parameter_count, load_best)


SIZES = (64, 256, 1024)


class Autoencoder(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(size, 128 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def image_scale(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x * std + mean).clamp(0, 1)


def normalize(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std


def corrupt(x, sigma, generator=None):
    if generator is None:
        noise = torch.randn_like(x)
    else:
        noise = torch.randn(x.shape, generator=generator).to(x.device)
    return (x + sigma * noise).clamp(0, 1)


def ae_epoch(model, batches, device, optimizer=None, sigma=0, seed=SEED + 40):
    model.train(optimizer is not None)
    generator = None if optimizer is not None else torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total = 0
    for x, _ in batches:
        clean = image_scale(x.to(device))
        inputs = corrupt(clean, sigma, generator) if sigma else clean
        with torch.set_grad_enabled(optimizer is not None):
            output = model(inputs)
            loss = (output - clean).square().mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(x)
        total += len(x)
    return total_loss / total


def train_ae(size, name, train_batches, val_batches, device, epochs, sigma=0):
    model = Autoencoder(size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best = float('inf')
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_mse = ae_epoch(model, train_batches, device, optimizer, sigma)
        val_mse = ae_epoch(model, val_batches, device, sigma=sigma)
        history.append({'epoch': epoch, 'train_mse': train_mse,
                        'val_mse': val_mse, 'seconds': time.perf_counter() - start})
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                        'val_mse': val_mse}, CHECKPOINTS / f'{name}.pt')
        print(f'{name} epoch={epoch:02d} train_mse={train_mse:.5f} '
              f'val_mse={val_mse:.5f}', flush=True)
    pd.DataFrame(history).to_csv(RESULTS / f'{name}_history.csv', index=False)
    return {'model': name, 'bottleneck': size, 'parameters': parameter_count(model),
            'best_val_mse': best, 'best_epoch': best_epoch,
            'training_seconds': sum(r['seconds'] for r in history)}


@torch.no_grad()
def features(model, batches, device):
    model.eval()
    vectors, labels = [], []
    for x, y in batches:
        vectors.append(model.encoder(image_scale(x.to(device))).cpu())
        labels.append(y)
    return torch.cat(vectors), torch.cat(labels)


def train_probe(name, size, train_batches, val_batches, device, epochs):
    ae = load_best(Autoencoder(size), name, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = features(ae, train_batches, device)
    val_x, val_y = features(ae, val_batches, device)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    batches = DataLoader(TensorDataset(train_x, train_y), batch_size=512, shuffle=True)
    probe = nn.Linear(size, 10).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        probe.train()
        for x, y in batches:
            scores = probe(x.to(device))
            loss = nn.functional.cross_entropy(scores, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(val_x.to(device)).argmax(1).cpu()
            accuracy = 100 * (pred == val_y).float().mean().item()
        history.append({'epoch': epoch, 'val_accuracy': accuracy})
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            torch.save({'state_dict': probe.state_dict(), 'mean': mean, 'std': std,
                        'bottleneck': size, 'epoch': epoch},
                       CHECKPOINTS / f'{name}_probe.pt')
    pd.DataFrame(history).to_csv(RESULTS / f'{name}_probe_history.csv', index=False)
    print(f'{name} frozen probe: validation accuracy={best:.2f}%', flush=True)
    return {'model': name, 'bottleneck': size,
            'best_val_probe_accuracy': best, 'best_epoch': best_epoch}


def plot_losses(names):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        path = RESULTS / f'{name}_history.csv'
        if path.exists():
            hist = pd.read_csv(path)
            ax.plot(hist.epoch, hist.val_mse, label=name)
    ax.set(xlabel='Epoch', ylabel='Validation reconstruction MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'q3_reconstruction_curves.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['all', 'clean', 'denoise', 'probes'], default='all')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--probe-epochs', type=int, default=15)
    p.add_argument('--sigma', type=float, default=0.15)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if not 0 < args.sigma <= 1:
        raise ValueError('sigma must be between 0 and 1')

    device = setup()
    train, val, _ = datasets_from_saved_split()
    tr = loader(train, args.batch_size, True, args.workers)
    tr_fixed = loader(train, args.batch_size, False, args.workers)
    va = loader(val, args.batch_size, False, args.workers)
    summary_path = RESULTS / 'q3_autoencoders.csv'
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    if args.stage in ('all', 'clean'):
        for size in SIZES:
            name = f'q3_ae_{size}'
            if (CHECKPOINTS / f'{name}.pt').exists() and name in summary.get('model', pd.Series(dtype=str)).values:
                print(f'{name} already trained, skipping')
                continue
            setup(SEED)
            row = train_ae(size, name, tr, va, device, args.epochs)
            summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
            summary = summary.drop_duplicates('model', keep='last')
            summary.to_csv(summary_path, index=False)

    if not summary_path.exists() or not all(f'q3_ae_{s}' in summary.model.values for s in SIZES):
        raise FileNotFoundError('First train all three clean autoencoders with --stage clean')

    clean_summary = summary[summary.model.isin([f'q3_ae_{s}' for s in SIZES])]
    chosen = clean_summary.loc[clean_summary.best_val_mse.idxmin()]
    size = int(chosen.bottleneck)
    print(f'Selected by validation reconstruction MSE: bottleneck={size}')
    selected_path = RESULTS / 'q3_selection.json'
    if selected_path.exists():
        saved = json.loads(selected_path.read_text())
        if saved['bottleneck'] != size or abs(saved['sigma'] - args.sigma) > 1e-12:
            raise ValueError('Existing selection uses a different bottleneck or sigma')
    else:
        selected_path.write_text(json.dumps({'bottleneck': size, 'sigma': args.sigma}, indent=2))

    if args.stage in ('all', 'denoise'):
        name = 'q3_dae'
        if (CHECKPOINTS / f'{name}.pt').exists() and name in summary.model.values:
            print(f'{name} already trained, skipping')
        else:
            setup(SEED + 1)
            row = train_ae(size, name, tr, va, device, args.epochs, args.sigma)
            summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
            summary = summary.drop_duplicates('model', keep='last')
            summary.to_csv(summary_path, index=False)

    if args.stage in ('all', 'probes'):
        probe_path = RESULTS / 'q3_probes.csv'
        probes = pd.read_csv(probe_path) if probe_path.exists() else pd.DataFrame()
        for name, dim in [(f'q3_ae_{s}', s) for s in SIZES] + [('q3_dae', size)]:
            if not (CHECKPOINTS / f'{name}.pt').exists():
                raise FileNotFoundError(f'Missing model: {name}; run --stage denoise first')
            if (CHECKPOINTS / f'{name}_probe.pt').exists() and name in probes.get('model', pd.Series(dtype=str)).values:
                print(f'{name} probe already trained, skipping')
                continue
            setup(SEED + 2)
            row = train_probe(name, dim, tr_fixed, va, device, args.probe_epochs)
            probes = pd.concat([probes, pd.DataFrame([row])], ignore_index=True)
            probes = probes.drop_duplicates('model', keep='last')
            probes.to_csv(probe_path, index=False)

    plot_losses([f'q3_ae_{s}' for s in SIZES] + ['q3_dae'])
    print('\nReconstruction summary:')
    print(summary.to_string(index=False))
    if (RESULTS / 'q3_probes.csv').exists():
        print('\nFrozen probe validation results:')
        print(pd.read_csv(RESULTS / 'q3_probes.csv').to_string(index=False))
    print('\nRun q3_test.py once after training and validation selection are complete.')


if __name__ == '__main__':
    main()
