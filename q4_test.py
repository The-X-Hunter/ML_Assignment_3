import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.svm import SVC

from common import SEED, MEAN, STD, RESULTS, CHECKPOINTS, FIGURES, setup, datasets_from_saved_split, loader, load_best
from q3_train import Autoencoder
from q4_train import get_features

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def show_image(dataset, index, ax):
    image, _ = dataset[int(index)]
    rgb = image.numpy().transpose(1, 2, 0) * np.asarray(STD) + np.asarray(MEAN)
    ax.imshow(np.clip(rgb, 0, 1))
    ax.axis('off')


def inspect_pair(pair, train_data, test_data, x_train, y_train, x_test, y_test,
                 kernel, C, output_rows):
    a, b = pair
    train_idx = np.flatnonzero((y_train == a) | (y_train == b))
    test_idx = np.flatnonzero((y_test == a) | (y_test == b))
    model = SVC(C=C, kernel=kernel, gamma='scale', degree=3, cache_size=1000)
    model.fit(x_train[train_idx], y_train[train_idx])
    scores = model.decision_function(x_test[test_idx])
    closest = test_idx[np.argsort(np.abs(scores))[:4]]
    support = train_idx[model.support_]
    support_scores = model.decision_function(x_train[support])
    nearest_support = support[np.argsort(np.abs(support_scores))[:4]]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for j, index in enumerate(nearest_support):
        show_image(train_data, index, axes[0, j])
        val = model.decision_function(x_train[index:index+1])[0]
        axes[0, j].set_title(f'SV: {CLASSES[y_train[index]]}\nscore {val:.3f}', fontsize=9)
        output_rows.append({'pair': f'{CLASSES[a]} / {CLASSES[b]}', 'source': 'train_support_vector',
                            'index': int(index), 'true_class': CLASSES[y_train[index]], 'decision_score': float(val)})
    for j, index in enumerate(closest):
        show_image(test_data, index, axes[1, j])
        val = model.decision_function(x_test[index:index+1])[0]
        pred = model.predict(x_test[index:index+1])[0]
        axes[1, j].set_title(f'Test: {CLASSES[y_test[index]]}\npred {CLASSES[pred]}, {val:.3f}', fontsize=9)
        output_rows.append({'pair': f'{CLASSES[a]} / {CLASSES[b]}', 'source': 'test_near_boundary',
                            'index': int(index), 'true_class': CLASSES[y_test[index]], 'decision_score': float(val)})
    fig.suptitle(f'{CLASSES[a]} vs {CLASSES[b]}: {len(model.support_)} pairwise support vectors')
    fig.tight_layout()
    fig.savefig(FIGURES / f'q4_boundary_{CLASSES[a]}_{CLASSES[b]}.png', dpi=180)
    plt.close(fig)
    return len(model.support_)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    output = RESULTS / 'q4_final_test.csv'
    if output.exists():
        raise FileExistsError('Q4 test results already exist; do not rerun or tune using test results')

    config = json.loads((RESULTS / 'q4_selection.json').read_text())
    objects = joblib.load(CHECKPOINTS / 'q4_models.joblib')
    if config['representation'] != 'q3_dae' or objects['representation'] != 'q3_dae':
        raise ValueError('Unexpected representation')

    device = setup()
    encoder = load_best(Autoencoder(int(config['bottleneck'])), 'q3_dae', device)
    train_data, _, test_data = datasets_from_saved_split()
    all_train_x, all_train_y = get_features(encoder, train_data, device, args.batch_size, args.workers)
    test_x, test_y = get_features(encoder, test_data, device, args.batch_size, args.workers)
    idx = objects['subset_indices']
    train_x = objects['scaler'].transform(all_train_x[idx])
    train_y = all_train_y[idx]
    if not np.array_equal(train_y, objects['train_classes']):
        raise ValueError('Saved training subset does not match the current CIFAR-10 split')
    x_test = objects['scaler'].transform(test_x)

    svm = objects['svm']
    clustering = objects['clusterer']
    pred = svm.predict(x_test)
    clusters = clustering.predict(x_test)
    cluster_labels = objects['mapping'][clusters]
    svm_accuracy = 100 * accuracy_score(test_y, pred)
    cluster_accuracy = 100 * accuracy_score(test_y, cluster_labels)
    ari = adjusted_rand_score(test_y, clusters)
    nmi = normalized_mutual_info_score(test_y, clusters)
    pd.DataFrame([{'svm_test_accuracy': svm_accuracy,
                   'cluster_majority_test_accuracy': cluster_accuracy,
                   'cluster_test_ARI': ari, 'cluster_test_NMI': nmi,
                   'svm_cluster_disagreement_fraction': np.mean(pred != cluster_labels),
                   'representation': 'q3_dae', 'svm_kernel': config['kernel'],
                   'svm_C': config['C'], 'clusters': config['clusters'],
                   'training_subset': config['train_subset']}]).to_csv(output, index=False)

    contingency = pd.crosstab(pd.Series(test_y, name='true_class'),
                              pd.Series(clusters, name='cluster'), dropna=False)
    contingency.index = [CLASSES[int(i)] for i in contingency.index]
    contingency.to_csv(RESULTS / 'q4_cluster_contingency.csv')

    matrix = confusion_matrix(test_y, pred, labels=np.arange(10))
    pairs = sorted([(matrix[a, b] + matrix[b, a], a, b)
                    for a in range(10) for b in range(a+1, 10)], reverse=True)[:3]
    pair_results = []
    boundary_rows = []
    for errors, a, b in pairs:
        keep = (test_y == a) | (test_y == b)
        count = int(np.sum(keep))
        num_support = inspect_pair((a, b), train_data, test_data, train_x,
                                   train_y, x_test, test_y, config['kernel'], config['C'], boundary_rows)
        pair_results.append({'class_a': CLASSES[a], 'class_b': CLASSES[b],
                             'svm_mutual_confusions': int(errors),
                             'pair_test_examples': count,
                             'svm_pair_test_accuracy': 100 * np.mean(pred[keep] == test_y[keep]),
                             'cluster_pair_test_accuracy': 100 * np.mean(cluster_labels[keep] == test_y[keep]),
                             'svm_cluster_pair_disagreement': np.mean(pred[keep] != cluster_labels[keep]),
                             'pairwise_support_vectors': num_support})
    pd.DataFrame(pair_results).to_csv(RESULTS / 'q4_class_pairs.csv', index=False)
    pd.DataFrame(boundary_rows).to_csv(RESULTS / 'q4_boundary_examples.csv', index=False)

    print('\nFinal test results:')
    print(pd.read_csv(output).to_string(index=False))
    print('\nThree most confused class pairs:')
    print(pd.DataFrame(pair_results).to_string(index=False))
    print('\nThe test labels were used only for final evaluation and example analysis.')


if __name__ == '__main__':
    main()
