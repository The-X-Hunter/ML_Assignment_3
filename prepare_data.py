import json
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from common import DATA, SPLITS, SEED, setup


def main():
    setup()
    training = CIFAR10(DATA, train=True, download=True)
    CIFAR10(DATA, train=False, download=True)
    targets = np.asarray(training.targets)
    path = SPLITS / 'cifar10_seed_260003.npz'
    if path.exists():
        split = np.load(path)
        tr, va = split['train_idx'], split['val_idx']
    else:
        tr, va = train_test_split(np.arange(len(targets)), train_size=45000,
                                  test_size=5000, stratify=targets, random_state=SEED)
        np.savez(path, train_idx=tr, val_idx=va)
    assert len(tr) == 45000 and len(va) == 5000
    assert len(set(tr) & set(va)) == 0
    assert np.all(np.bincount(targets[tr], minlength=10) == 4500)
    assert np.all(np.bincount(targets[va], minlength=10) == 500)
    print(f'Saved split: {path}')
    print('Train:', len(tr), 'Validation:', len(va), 'Test: 10000')
    (SPLITS / 'split_metadata.json').write_text(json.dumps({
        'seed': SEED, 'train': 45000, 'validation': 5000, 'test': 10000,
        'method': 'sklearn train_test_split with stratify=targets'
    }, indent=2))

if __name__ == '__main__':
    main()
