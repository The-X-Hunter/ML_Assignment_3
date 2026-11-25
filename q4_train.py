import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.optimize import linear_sum_assignment

from common import SEED, RESULTS, CHECKPOINTS, SPLITS, FIGURES, setup, datasets_from_saved_split, loader, load_best
from q3_train import Autoencoder, features


def get_features(model, data, device, batch_size, workers):
    x, y = features(model, loader(data, batch_size, False, workers), device)
    return x.numpy().astype('float32'), y.numpy()


def training_subset(labels, size):
    if size % 10 or size < 100:
        raise ValueError('Subset size must be divisible by 10 and at least 100')
    path = SPLITS / f'q4_subset_{size}.npy'
    if path.exists():
        idx = np.load(path)
    else:
        rng = np.random.default_rng(SEED + 400)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(labels == c), size // 10, replace=False)
            for c in range(10)
        ])
        rng.shuffle(idx)
        np.save(path, idx)
    if len(idx) != size or len(np.unique(idx)) != size or not np.all(np.bincount(labels[idx], minlength=10) == size // 10):
        raise ValueError('Saved Q4 subset does not match this training split')
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()

    device = setup()
    if (RESULTS / 'q4_selection.json').exists() or (CHECKPOINTS / 'q4_models.joblib').exists():
        raise FileExistsError('Q4 models already selected; do not rerun selection after test evaluation')

    q3 = json.loads((RESULTS / 'q3_selection.json').read_text())
    size = int(q3['bottleneck'])
    probe = pd.read_csv(RESULTS / 'q3_probes.csv')
    best = probe.loc[probe.best_val_probe_accuracy.idxmax(), 'model']
    if best != 'q3_dae':
        raise ValueError(f'The validation-selected representation is {best}, not q3_dae')
    encoder = load_best(Autoencoder(size), 'q3_dae', device)
    train, val, _ = datasets_from_saved_split()
    train_x, train_y = get_features(encoder, train, device, args.batch_size, args.workers)
    val_x, val_y = get_features(encoder, val, device, args.batch_size, args.workers)
    idx = training_subset(train_y, args.subset)
    x_train, y_train = train_x[idx], train_y[idx]

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(val_x)

    records = []
    Cs = (0.01, 0.1, 1, 10, 100)
    kernels = ('linear', 'rbf', 'poly')
    chosen = None
    best_val = -1
    for kernel in kernels:
        for C in Cs:
            svm = SVC(C=C, kernel=kernel, gamma='scale', degree=3,
                      decision_function_shape='ovo', cache_size=1000)
            svm.fit(x_train, y_train)
            tr_acc = 100 * accuracy_score(y_train, svm.predict(x_train))
            va_acc = 100 * accuracy_score(val_y, svm.predict(x_val))
            fraction = len(svm.support_) / len(y_train)
            records.append({'kernel': kernel, 'C': C, 'train_accuracy': tr_acc,
                            'val_accuracy': va_acc, 'support_vector_fraction': fraction,
                            'support_vectors': len(svm.support_)})
            print(f'{kernel:6s} C={C:6g} train={tr_acc:6.2f}% val={va_acc:6.2f}% support={fraction:.3f}', flush=True)
            if va_acc > best_val:
                best_val, chosen = va_acc, (kernel, C)

    sweep = pd.DataFrame(records)
    sweep.to_csv(RESULTS / 'q4_svm_sweep.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kernel in kernels:
        part = sweep[sweep.kernel == kernel]
        axes[0].semilogx(part.C, part.val_accuracy, marker='o', label=kernel)
        axes[1].semilogx(part.C, part.support_vector_fraction, marker='o', label=kernel)
    axes[0].set(xlabel='C', ylabel='Validation accuracy (%)')
    axes[1].set(xlabel='C', ylabel='Support vector fraction')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'q4_svm_sweep.png', dpi=180)
    plt.close(fig)

    kernel, C = chosen
    svm = SVC(C=C, kernel=kernel, gamma='scale', degree=3,
              decision_function_shape='ovo', cache_size=1000)
    svm.fit(x_train, y_train)

    clustering_records = []
    rng = np.random.default_rng(SEED + 401)
    eval_idx = rng.choice(len(x_val), size=min(1500, len(x_val)), replace=False)
    best_score = -float('inf')
    selected_k = None
    for k in (10, 20, 40):
        clusterer = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=1024,
                                   n_init=5, max_iter=200)
        clusterer.fit(x_train)
        assigned = clusterer.predict(x_val)
        score = silhouette_score(x_val[eval_idx], assigned[eval_idx]) if len(np.unique(assigned[eval_idx])) > 1 else -1
        clustering_records.append({'method': 'MiniBatchKMeans', 'clusters': k,
                                   'validation_silhouette': score})
        print(f'Clustering k={k} validation silhouette={score:.4f}', flush=True)
        if score > best_score:
            best_score, selected_k = score, k
    pd.DataFrame(clustering_records).to_csv(RESULTS / 'q4_cluster_selection.csv', index=False)

    clusterer = MiniBatchKMeans(n_clusters=selected_k, random_state=SEED,
                               batch_size=1024, n_init=5, max_iter=200)
    clusterer.fit(x_train)
    train_cluster = clusterer.predict(x_train)
    counts = np.zeros((selected_k, 10), dtype=int)
    np.add.at(counts, (train_cluster, y_train), 1)
    assignment = linear_sum_assignment(counts, maximize=True)
    mapping = counts.argmax(axis=1)
    mapping[assignment[0]] = assignment[1]

    joblib.dump({'scaler': scaler, 'svm': svm, 'clusterer': clusterer,
                 'mapping': mapping, 'subset_indices': idx, 'train_classes': y_train,
                 'representation': 'q3_dae'}, CHECKPOINTS / 'q4_models.joblib')
    config = {'representation': 'q3_dae', 'bottleneck': size,
              'train_subset': args.subset, 'seed': SEED + 400,
              'kernel': kernel, 'C': C, 'best_svm_validation_accuracy': best_val,
              'clustering': 'MiniBatchKMeans', 'clusters': selected_k,
              'best_validation_silhouette': best_score,
              'svm_support_vectors': len(svm.support_)}
    (RESULTS / 'q4_selection.json').write_text(json.dumps(config, indent=2))
    print('\nSelected configuration:')
    print(json.dumps(config, indent=2))
    print('Run q4_test.py once. Do not change model choices after inspecting test results.')


if __name__ == '__main__':
    main()
