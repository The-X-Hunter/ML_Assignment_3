import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from common import (SEED, MEAN, STD, RESULTS, FIGURES, MLP, setup,
                    datasets_from_saved_split, loader, fit, parameter_count,
                    plot_histories)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 32, 64, 128]
        for i in range(3):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 768),
            nn.ReLU(),
            nn.Linear(768, 10)
        )

    def forward(self, x):
        return self.net(x)


class ImageCondition(Dataset):
    def __init__(self, dataset, condition):
        self.dataset = dataset
        self.condition = condition
        self.permutation = torch.from_numpy(
            np.random.default_rng(SEED + 31).permutation(1024).copy()
        ).long()
        self.mean = torch.tensor(MEAN)[:, None, None]
        self.std = torch.tensor(STD)[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.condition == 'permuted':
            x = x.reshape(3, 1024)[:, self.permutation].reshape(3, 32, 32)
        elif self.condition == 'grayscale':
            rgb = x * self.std + self.mean
            gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
            x = (gray.unsqueeze(0).repeat(3, 1, 1) - self.mean) / self.std
        return x, y


def create_model(kind):
    return CNN() if kind == 'cnn' else MLP((512, 256), batchnorm=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--condition', choices=['all', 'clean', 'permuted', 'grayscale'], default='all')
    args = p.parse_args()

    device = setup()
    train, val, _ = datasets_from_saved_split()
    conditions = ['clean', 'permuted', 'grayscale'] if args.condition == 'all' else [args.condition]

    summaries = []
    for condition in conditions:
        kinds = ['cnn'] if condition == 'clean' else ['mlp', 'cnn']
        train_batches = loader(ImageCondition(train, condition), args.batch_size, True, args.workers)
        val_batches = loader(ImageCondition(val, condition), args.batch_size, False, args.workers)
        for kind in kinds:
            name = f'q2_{kind}_{condition}'
            setup(SEED)
            model = create_model(kind)
            print(f'Training {name}: {parameter_count(model)} parameters', flush=True)
            _, summary = fit(model, train_batches, val_batches, name, args.epochs, device)
            summary['condition'] = condition
            summary['architecture'] = kind
            summaries.append(summary)

        old_path = RESULTS / 'q2_training_summary.csv'
        old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
        combined = pd.concat([old, pd.DataFrame(summaries)], ignore_index=True)
        combined = combined.drop_duplicates(subset='model', keep='last')
        combined.to_csv(old_path, index=False)

    histories = {}
    plain_history = RESULTS / 'q1_batchnorm_history.csv'
    if plain_history.exists():
        histories['mlp_clean'] = pd.read_csv(plain_history)
    for name in ['q2_cnn_clean', 'q2_mlp_permuted', 'q2_cnn_permuted',
                 'q2_mlp_grayscale', 'q2_cnn_grayscale']:
        path = RESULTS / f'{name}_history.csv'
        if path.exists():
            histories[name.replace('q2_', '')] = pd.read_csv(path)
    if histories:
        plot_histories(histories, FIGURES / 'q2_training_curves.png')
    print('\nValidation summary:')
    print(pd.read_csv(RESULTS / 'q2_training_summary.csv').to_string(index=False))
    print('\nDo not use test data for model selection. Run q2_test.py only after choices are frozen.')


if __name__ == '__main__':
    main()
