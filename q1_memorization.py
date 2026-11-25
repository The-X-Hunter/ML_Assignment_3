import argparse
import time
import torch
import pandas as pd
from common import (setup, datasets_from_saved_split, loader, MLP,
                    NoisyLabels, memorization_scores, epoch, FIGURES, RESULTS,
                    CHECKPOINTS, plot_histories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--noise', type=float, default=0.20)
    args = parser.parse_args()
    assert 0 < args.noise < 1
    device = setup()
    tr, va, _ = datasets_from_saved_split()
    noisy = NoisyLabels(tr, fraction=args.noise)
    training = loader(noisy, args.batch_size, True, args.workers)
    validation = loader(va, args.batch_size, False, args.workers)
    model = MLP(hidden=(1024, 512)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    records = []
    best_val = -1
    for i in range(1, args.epochs+1):
        start = time.perf_counter()
        train_metrics = epoch(model, training, device, opt)
        val_metrics = epoch(model, validation, device)
        scores = memorization_scores(model, noisy, device, args.batch_size, args.workers)
        row = {'epoch': i, 'train_loss': train_metrics['loss'],
               'train_accuracy': train_metrics['accuracy'],
               'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'],
               **scores, 'seconds': time.perf_counter()-start}
        records.append(row)
        if row['val_accuracy'] > best_val:
            best_val = row['val_accuracy']
            torch.save({'state_dict': model.state_dict(), 'epoch': i,
                        'val_accuracy': best_val}, CHECKPOINTS / 'q1_memorization.pt')
        print(f"epoch={i:03d} val={row['val_accuracy']:.2f}% "
              f"clean={scores['clean_train_accuracy']:.2f}% "
              f"fit_wrong={scores['corrupted_label_fit']:.2f}%", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(RESULTS / 'q1_memorization_history.csv', index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for col in ('clean_train_accuracy', 'corrupted_label_fit',
                'corrupted_subset_true_accuracy', 'val_accuracy'):
        ax.plot(df.epoch, df[col], label=col.replace('_',' '))
    ax.set(xlabel='Epoch', ylabel='Accuracy (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'q1_memorization.png', dpi=180)
    plt.close(fig)
    print('Finished. Use the trajectory to determine whether memorization actually begins.')

if __name__ == '__main__':
    main()
