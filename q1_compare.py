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
    device = setup()
    tr, va, _ = datasets_from_saved_split()
    train_loader = loader(tr, args.batch_size, True, args.workers)
    val_loader = loader(va, args.batch_size, False, args.workers)
    configs = {'plain': False, 'batchnorm': True}
    histories, rows = {}, []
    for name, bn in configs.items():
        setup(SEED)
        model = MLP(hidden=(512, 256), batchnorm=bn)
        history, summary = fit(model, train_loader, val_loader,
                               f'q1_{name}', args.epochs, device)
        histories[name] = history
        rows.append(summary)
    pd.DataFrame(rows).to_csv(RESULTS / 'q1_comparison.csv', index=False)
    plot_histories(histories, FIGURES / 'q1_comparison.png')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Hypothesis: batch normalization changes optimization at near-matched capacity; '
          'do not interpret this experiment alone as proof of spatial inductive bias.')

if __name__ == '__main__':
    main()
