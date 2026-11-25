import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from common import (SEED, RESULTS, CHECKPOINTS, FIGURES, setup,
                    datasets_from_saved_split, loader, load_best)
from q2_train import CNN
from q3_train import Autoencoder, SIZES, image_scale, normalize, corrupt


@torch.no_grad()
def reconstruction_mse(model, batches, device, sigma=0):
    model.eval()
    generator = torch.Generator().manual_seed(SEED + 40)
    total = 0.0
    n = 0
    for x, _ in batches:
        clean = image_scale(x.to(device))
        inputs = corrupt(clean, sigma, generator) if sigma else clean
        pred = model(inputs)
        total += (pred - clean).square().sum().item()
        n += clean.numel()
    return total / n


@torch.no_grad()
def classifier_results(cnn, batches, clean_ae, dae, device, sigma):
    cnn.eval()
    clean_ae.eval()
    dae.eval()
    generator = torch.Generator().manual_seed(SEED + 40)
    correct = {k: 0 for k in ('clean', 'corrupted', 'clean_ae', 'denoised')}
    confidence = {k: 0.0 for k in correct}
    total = 0
    for x, y in batches:
        clean = image_scale(x.to(device))
        noisy = corrupt(clean, sigma, generator)
        inputs = {
            'clean': clean,
            'corrupted': noisy,
            'clean_ae': clean_ae(noisy),
            'denoised': dae(noisy)
        }
        labels = y.to(device)
        total += len(y)
        for name, images in inputs.items():
            probs = cnn(normalize(images)).softmax(1)
            conf, predicted = probs.max(1)
            correct[name] += (predicted == labels).sum().item()
            confidence[name] += conf.sum().item()
    return [{'input': key, 'test_accuracy': 100 * correct[key] / total,
             'mean_confidence': confidence[key] / total}
            for key in correct]


@torch.no_grad()
def probe_accuracy(ae, probe_checkpoint, batches, device):
    ae.eval()
    size = int(probe_checkpoint['bottleneck'])
    probe = nn.Linear(size, 10).to(device)
    probe.load_state_dict(probe_checkpoint['state_dict'])
    probe.eval()
    mean = probe_checkpoint['mean'].to(device)
    std = probe_checkpoint['std'].to(device)
    correct = total = 0
    for x, y in batches:
        latent = ae.encoder(image_scale(x.to(device)))
        pred = probe((latent - mean) / std).argmax(1)
        correct += (pred == y.to(device)).sum().item()
        total += len(y)
    return 100 * correct / total


@torch.no_grad()
def reconstruction_figure(batches, clean_ae, dae, device, sigma):
    clean_ae.eval()
    dae.eval()
    x, _ = next(iter(batches))
    clean = image_scale(x[:6].to(device))
    generator = torch.Generator().manual_seed(SEED + 40)
    noisy = corrupt(clean, sigma, generator)
    clean_out = clean_ae(noisy)
    denoised = dae(noisy)
    names = ['Original', 'Corrupted', 'Clean AE', 'Denoising AE']
    groups = [clean, noisy, clean_out, denoised]
    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for i, (label, images) in enumerate(zip(names, groups)):
        for j, image in enumerate(images):
            axes[i, j].imshow(image.cpu().permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(label, rotation=0, labelpad=45, va='center')
    fig.tight_layout()
    fig.savefig(FIGURES / 'q3_reconstructions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    device = setup()
    output = RESULTS / 'q3_final_test.csv'
    if output.exists():
        raise FileExistsError('Q3 test already evaluated; do not tune using the test set')
    selection = RESULTS / 'q3_selection.json'
    if not selection.exists():
        raise FileNotFoundError('Run q3_train.py first')
    config = json.loads(selection.read_text())
    chosen = int(config['bottleneck'])
    sigma = float(config['sigma'])
    names = [f'q3_ae_{s}' for s in SIZES] + ['q3_dae']
    needed = names + [f'{name}_probe' for name in names] + ['q2_cnn_clean']
    missing = [name for name in needed if not (CHECKPOINTS / f'{name}.pt').exists()]
    if missing:
        raise FileNotFoundError('Missing checkpoints: ' + ', '.join(missing))

    _, _, test = datasets_from_saved_split()
    batches = loader(test, args.batch_size, False, args.workers)
    models = {}
    for size in SIZES:
        models[f'q3_ae_{size}'] = load_best(Autoencoder(size), f'q3_ae_{size}', device)
    models['q3_dae'] = load_best(Autoencoder(chosen), 'q3_dae', device)
    clean_ae = models[f'q3_ae_{chosen}']
    dae = models['q3_dae']
    cnn = load_best(CNN(), 'q2_cnn_clean', device)

    reconstruction = []
    for name, model in models.items():
        mse = reconstruction_mse(model, batches, device)
        reconstruction.append({'model': name, 'bottleneck': chosen if name == 'q3_dae' else int(name.split('_')[-1]),
                               'test_clean_reconstruction_mse': mse})
    pd.DataFrame(reconstruction).to_csv(RESULTS / 'q3_reconstruction_test.csv', index=False)

    rows = classifier_results(cnn, batches, clean_ae, dae, device, sigma)
    pd.DataFrame(rows).to_csv(output, index=False)
    representation = []
    for name, model in models.items():
        checkpoint = torch.load(CHECKPOINTS / f'{name}_probe.pt', map_location='cpu', weights_only=True)
        accuracy = probe_accuracy(model, checkpoint, batches, device)
        mse = next(row['test_clean_reconstruction_mse'] for row in reconstruction if row['model'] == name)
        representation.append({'model': name, 'bottleneck': int(checkpoint['bottleneck']),
                               'test_reconstruction_mse': mse, 'test_probe_accuracy': accuracy})
    pd.DataFrame(representation).to_csv(RESULTS / 'q3_representations_test.csv', index=False)

    reconstruction_figure(batches, clean_ae, dae, device, sigma)
    print('\nFixed CNN on clean, corrupted and reconstructed images:')
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nFrozen-encoder linear probes:')
    print(pd.DataFrame(representation).to_string(index=False))
    print('\nClean autoencoder selected using validation MSE:', chosen)


if __name__ == '__main__':
    main()
