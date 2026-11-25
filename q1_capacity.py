import argparse
import json

import pandas as pd

from common import (setup, datasets_from_saved_split, loader, MLP, fit,
                    plot_histories, RESULTS, FIGURES, SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    device = setup(SEED)
    train, val, _ = datasets_from_saved_split()
    train_loader = loader(train, args.batch_size, True, args.workers)
    val_loader = loader(val, args.batch_size, False, args.workers)

    model = MLP(hidden=(1024, 512), batchnorm=False)
    history, summary = fit(model, train_loader, val_loader,
                           'q1_wide', args.epochs, device)

    comparison_path = RESULTS / 'q1_comparison.csv'
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[comparison['model'] != 'q1_wide']
    else:
        rows = []
        for name in ('q1_plain', 'q1_batchnorm'):
            summary_path = RESULTS / f'{name}_summary.json'
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text()))
        comparison = pd.DataFrame(rows)

    comparison = pd.concat([comparison, pd.DataFrame([summary])], ignore_index=True)
    comparison.to_csv(comparison_path, index=False)

    histories = {}
    for name in ('plain', 'batchnorm', 'wide'):
        path = RESULTS / f'q1_{name}_history.csv'
        if path.exists():
            histories[name] = pd.read_csv(path)
    plot_histories(histories, FIGURES / 'q1_comparison.png')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
