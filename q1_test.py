"""Run only once model and epoch choices are frozen on validation data."""
import argparse
import pandas as pd
from common import (setup, datasets_from_saved_split, loader, MLP,
                    load_best, epoch, RESULTS, CHECKPOINTS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    device = setup()
    _, _, test = datasets_from_saved_split()
    test_loader = loader(test, args.batch_size, False, args.workers)
    rows = []
    configs = [('plain', (512, 256), False),
               ('batchnorm', (512, 256), True),
               ('wide', (1024, 512), False)]
    for name, hidden, bn in configs:
        if not (CHECKPOINTS / f'q1_{name}.pt').exists():
            print(f'Skipping q1_{name}: checkpoint missing')
            continue
        model = load_best(MLP(hidden=hidden, batchnorm=bn), f'q1_{name}', device)
        score = epoch(model, test_loader, device)
        rows.append({'model': f'q1_{name}', 'test_accuracy':score['accuracy'],
                     'test_cross_entropy':score['loss']})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'q1_final_test.csv', index=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
