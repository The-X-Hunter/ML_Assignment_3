import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from common import (RESULTS, FIGURES, CHECKPOINTS, setup,
                    datasets_from_saved_split, loader, load_best, parameter_count)
from q2_train import ImageCondition, create_model


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    true, predictions, confidence = [], [], []
    loss_sum = n = 0
    for x, y in batches:
        logits = model(x.to(device))
        probs = logits.softmax(dim=1)
        scores, pred = probs.max(dim=1)
        true.append(y.numpy())
        predictions.append(pred.cpu().numpy())
        confidence.append(scores.cpu().numpy())
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        n += len(y)
    true = np.concatenate(true)
    pred = np.concatenate(predictions)
    conf = np.concatenate(confidence)
    correct = true == pred
    return true, pred, conf, {
        'test_accuracy': 100 * correct.mean(),
        'test_cross_entropy': loss_sum / n,
        'mean_confidence': float(conf.mean()),
        'confidence_correct': float(conf[correct].mean()) if correct.any() else float('nan'),
        'confidence_wrong': float(conf[~correct].mean()) if (~correct).any() else float('nan')
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()

    configs = [('mlp', 'clean', 'q1_batchnorm'),
               ('cnn', 'clean', 'q2_cnn_clean'),
               ('mlp', 'permuted', 'q2_mlp_permuted'),
               ('cnn', 'permuted', 'q2_cnn_permuted'),
               ('mlp', 'grayscale', 'q2_mlp_grayscale'),
               ('cnn', 'grayscale', 'q2_cnn_grayscale')]
    missing = [name for _, _, name in configs if not (CHECKPOINTS / f'{name}.pt').exists()]
    if missing:
        raise FileNotFoundError('Missing checkpoints: ' + ', '.join(missing))
    output = RESULTS / 'q2_final_test.csv'
    if output.exists():
        raise FileExistsError(f'{output} already exists. Test evaluation should be run once after model selection.')

    device = setup()
    _, _, test = datasets_from_saved_split()
    rows, class_rows, error_rows = [], [], []
    names = test.dataset.classes if hasattr(test, 'dataset') else test.classes

    for kind, condition, checkpoint in configs:
        model = create_model(kind)
        model = load_best(model, checkpoint, device)
        true, pred, conf, metrics = evaluate(
            model, loader(ImageCondition(test, condition), args.batch_size, False, args.workers), device
        )
        correct = true == pred
        rows.append({'model': kind, 'condition': condition,
                     'parameters': parameter_count(model), **metrics})
        for cls in range(10):
            mask = true == cls
            class_rows.append({'model': kind, 'condition': condition,
                               'class': names[cls], 'test_accuracy': 100 * correct[mask].mean(),
                               'mean_confidence': float(conf[mask].mean()),
                               'wrong_predictions': int((~correct[mask]).sum())})
        pairs, counts = np.unique(
            np.stack([true[~correct], pred[~correct]], axis=1), axis=0, return_counts=True
        )
        for pair, count in zip(pairs, counts):
            error_rows.append({'model': kind, 'condition': condition,
                               'true_class': names[pair[0]], 'predicted_class': names[pair[1]],
                               'count': int(count)})
        print(f'{kind:3s} {condition:10s} accuracy={metrics["test_accuracy"]:.2f}% '
              f'confidence={metrics["mean_confidence"]:.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / 'q2_class_accuracy.csv', index=False)
    pd.DataFrame(error_rows).sort_values(
        ['model', 'condition', 'count'], ascending=[True, True, False]
    ).to_csv(RESULTS / 'q2_errors.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind in ['mlp', 'cnn']:
        subset = df[df.model == kind].set_index('condition').reindex(['clean','permuted','grayscale'])
        axes[0].plot(['clean','permuted','grayscale'], subset.test_accuracy, marker='o', label=kind)
        axes[1].plot(['clean','permuted','grayscale'], subset.mean_confidence, marker='o', label=kind)
    axes[0].set(ylabel='Test accuracy (%)', xlabel='Input condition')
    axes[1].set(ylabel='Mean winning probability', xlabel='Input condition')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'q2_test_comparison.png', dpi=180)
    plt.close(fig)
    print('\n', df.to_string(index=False))


if __name__ == '__main__':
    main()
