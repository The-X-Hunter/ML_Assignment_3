
# Machine Learning Assignment 3

**Course:** Machine Learning  
**Topic:** MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10  

## Overview

This repository contains the implementation and experimental results for Machine Learning Assignment 3.

The assignment investigates how different machine learning models learn from CIFAR-10 images, how well they generalize, and how their behavior changes under different experimental conditions.

The work is divided into four parts:

- **Q1:** MLP capacity, batch normalization, and memorization.
- **Q2:** CNN versus MLP on clean, spatially permuted, and grayscale images.
- **Q3:** Convolutional autoencoders, denoising, and learned representations.
- **Q4:** SVM classification and clustering using frozen autoencoder features.

All neural networks were trained from scratch. No pretrained models or external datasets were used.

## Environment Setup

The implementation uses Python 3.11.

Create and activate a Conda environment:

```bash
conda create -n ml_assignment3 python=3.11 -y
conda activate ml_assignment3
```

Install a PyTorch build compatible with your CUDA driver. For systems supporting the CUDA 12.1 build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The main libraries are PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, Matplotlib, and joblib.

## Project Structure

```text
ML_Assignment3/
├── common.py
├── prepare_data.py
├── q1_compare.py
├── q1_capacity.py
├── q1_memorization.py
├── q1_test.py
├── q2_train.py
├── q2_test.py
├── q3_train.py
├── q3_test.py
├── q4_train.py
├── q4_test.py
├── requirements.txt
├── README.md
├── data/
├── splits/
├── checkpoints/
├── results/
└── figures/
```

- `data/`: Downloaded CIFAR-10 dataset.
- `splits/`: Saved training and validation indices.
- `checkpoints/`: Trained model weights and fitted estimators.
- `results/`: Training histories, evaluation results, and model-selection records.
- `figures/`: Plots and visualizations used in the report.

All commands should be executed from the project directory.

## Dataset Preparation

We used CIFAR-10, which contains 60,000 RGB images of size 32 × 32 from 10 classes.

The original 50,000 training images were divided into a stratified training and validation split.

| Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Official test | 10,000 |

The random seed was derived from the numeric characters of the roll number:

`P26AI0003 → 260003`

Prepare the dataset:

```bash
python prepare_data.py
```

The generated split is saved and reused across the experiments.

Validation data was used for architecture selection, hyperparameter selection, and checkpoint selection. The official test set was used only after these choices were fixed.

## Q1: MLP Experiments

### Q1(a): Capacity and Batch Normalization

Three MLP configurations were compared:

| Model | Hidden Layers | Parameters | Test Accuracy |
|---|---|---:|---:|
| Plain MLP | 512, 256 | 1,707,274 | 52.16% |
| BatchNorm MLP | 512, 256 | 1,708,810 | 55.44% |
| Wider MLP | 1024, 512 | 3,676,682 | 53.25% |

The plain MLP and BatchNorm MLP have almost identical parameter counts, allowing a comparison of the effect of normalization.

The wider MLP was added to investigate whether increasing capacity improves generalization.

All three models used the same data split, optimizer settings, and 30-epoch training budget.

Run:

```bash
python q1_compare.py --epochs 30
python q1_capacity.py --epochs 30
python q1_test.py
```

The BatchNorm MLP achieved the highest accuracy. Increasing the number of parameters alone did not provide a comparable improvement.

### Q1(b): Memorization

We deliberately changed 20% of the training labels to incorrect classes and trained an MLP for 60 epochs.

During training, we recorded:

- Accuracy on correctly labeled training examples.
- Accuracy against deliberately incorrect training labels.
- Clean validation accuracy.

Run:

```bash
python q1_memorization.py --epochs 60
```

The highest validation accuracy was 47.84% at epoch 7.

By epoch 60, the model fitted 81.08% of the incorrect labels, while validation accuracy decreased to 42.36%.

The experiment shows that useful learning became visible early in training, while substantial memorization developed later.

Results are saved in `results/q1_*`, and the plots are stored in `figures/`.

## Q2: CNN vs MLP

The batch-normalized MLP from Q1 was reused and compared with a CNN of similar parameter count.

Three experiments were conducted:

1. Classification using clean CIFAR-10 images.
2. Retraining and evaluation using spatially permuted images.
3. Retraining and evaluation using grayscale images.

For spatial permutation, one fixed permutation was applied to all 1024 image positions. The same permutation was used for every image and all RGB channels.

For grayscale, the intensity channel was repeated three times to maintain the original input shape.

Run:

```bash
python q2_train.py --epochs 30
python q2_test.py
```

### Results

| Condition | MLP Test Accuracy | CNN Test Accuracy |
|---|---:|---:|
| Clean | 55.44% | 78.67% |
| Spatially permuted | 54.69% | 52.18% |
| Grayscale | 47.17% | 76.41% |

The CNN performed much better on clean images, but its advantage decreased sharply when spatial structure was disturbed.

Removing color information had a smaller effect on the CNN.

We also measured prediction confidence, class-wise accuracy, and confusion counts.

Results and figures are saved under `results/q2_*` and `figures/`.

## Q3: Convolutional Autoencoders

### Q3(a): Bottleneck Capacity

Three convolutional autoencoders were trained using bottleneck dimensions of 64, 256, and 1024.

The remaining architecture and training settings were kept comparable.

| Bottleneck | Parameters | Test Reconstruction MSE |
|---|---:|---:|
| 64 | 522,979 | 0.007778 |
| 256 | 1,309,603 | 0.003190 |
| 1024 | 4,456,099 | 0.002503 |

The 1024-dimensional autoencoder was selected because it achieved the lowest validation reconstruction error.

### Q3(b): Denoising

Gaussian noise with standard deviation 0.15 was used to corrupt images.

A denoising autoencoder was trained using corrupted images as inputs and clean images as targets.

The fixed CNN from Q2 was evaluated under four conditions.

| Input | CNN Test Accuracy |
|---|---:|
| Clean | 78.67% |
| Corrupted | 17.73% |
| Ordinary AE reconstruction | 54.43% |
| Denoising AE reconstruction | 52.47% |

Both autoencoders improved classification performance on corrupted inputs, although neither restored clean-image performance.

### Q3(c): Frozen Representations

The trained encoders were frozen, and linear classifiers were trained using their latent features.

| Representation | Test Probe Accuracy |
|---|---:|
| AE-64 | 37.42% |
| AE-256 | 43.09% |
| AE-1024 | 46.55% |
| Denoising AE-1024 | 48.01% |

The denoising encoder achieved the highest linear-probe accuracy despite not having the lowest clean-input reconstruction error.

This shows that reconstruction quality and class information do not always agree.

Run:

```bash
python q3_train.py --epochs 20 --probe-epochs 15
python q3_test.py
```

Results are stored in `results/q3_*`, with figures in `figures/`.

## Q4: SVM and Clustering

The 1024-dimensional denoising autoencoder from Q3 was used as the fixed feature extractor.

The encoder was selected using validation results and remained frozen throughout Q4.

### Q4(a): SVM Experiments

A fixed stratified subset of 5,000 training images was used to reduce kernel SVM training cost.

The main CIFAR-10 split used seed `260003`. Subset selection for Q4 used seed `260403` to maintain a separate, reproducible sampling sequence.

The same subset was used for every SVM configuration.

Three kernels were compared:

- Linear
- RBF
- Polynomial

Five regularization values were evaluated:

`C = 0.01, 0.1, 1, 10, 100`

This produced 15 SVM configurations.

For every configuration, we recorded training accuracy, validation accuracy, and the fraction of support vectors.

The best configuration was selected using validation accuracy.

| Setting | Selected Value |
|---|---|
| Kernel | RBF |
| C | 10 |
| Validation accuracy | 47.96% |
| Test accuracy | 46.70% |
| Support vectors | 4,905 / 5,000 |

### Q4(b): Difficult Class Pairs

Three class pairs with high mutual confusion were studied:

- Automobile and truck.
- Cat and dog.
- Bird and deer.

Training support vectors and test examples near binary SVM decision boundaries were examined, along with their decision scores.

### Q4(c): Clustering

MiniBatch K-means was applied to the same frozen representation without class labels during clustering.

Cluster counts of 10, 20, and 40 were compared using validation silhouette scores.

Ten clusters were selected.

Final results:

| Metric | Value |
|---|---:|
| Test ARI | 0.0510 |
| Test NMI | 0.1003 |
| Mapped cluster accuracy | 22.01% |
| SVM–cluster disagreement | 74.51% |

The clustering results showed weak alignment with the CIFAR-10 labels compared with the supervised SVM.

Run:

```bash
python q4_train.py --subset 5000
python q4_test.py
```

The SVM sweep, model-selection results, boundary examples, clustering results, and contingency table are saved under `results/q4_*`.

## Complete Execution Order

To reproduce all experiments, run:

```bash
python prepare_data.py

python q1_compare.py --epochs 30
python q1_capacity.py --epochs 30
python q1_memorization.py --epochs 60
python q1_test.py

python q2_train.py --epochs 30
python q2_test.py

python q3_train.py --epochs 20 --probe-epochs 15
python q3_test.py

python q4_train.py --subset 5000
python q4_test.py
```

The scripts reuse previously trained models where required:

- Q2 reuses the BatchNorm MLP from Q1.
- Q3 reuses the clean CNN from Q2.
- Q4 reuses the denoising autoencoder from Q3.

Run the commands from the project root directory.

## Saved Results

The scripts save numerical results automatically in `results/`.

Important files include:

```text
results/
├── q1_comparison.csv
├── q1_final_test.csv
├── q1_memorization_history.csv
├── q2_training_summary.csv
├── q2_final_test.csv
├── q2_class_accuracy.csv
├── q2_errors.csv
├── q3_autoencoders.csv
├── q3_probes.csv
├── q3_reconstruction_test.csv
├── q3_final_test.csv
├── q3_representations_test.csv
├── q4_svm_sweep.csv
├── q4_cluster_selection.csv
├── q4_final_test.csv
├── q4_class_pairs.csv
├── q4_boundary_examples.csv
└── q4_cluster_contingency.csv
```

Additional training histories and model-selection summaries are also saved.

The trained model weights are stored in `checkpoints/`, and the figures are stored in `figures/`.

## Reproducibility

The experiments use:

- A fixed stratified training and validation split.
- Saved split indices.
- Fixed random seeds.
- Validation-based model selection.
- Saved model checkpoints and training histories.
- Consistent evaluation conditions.
- No pretrained models or external data.
- No test-set-based hyperparameter selection.

The supplied result files correspond to the reported runs. Minor numerical differences may occur across hardware and software environments.

## Report

The final report follows a two-column CVPR-style format and covers Q1–Q4.

It includes model descriptions, controlled experiments, measured results, figures, interpretations, and limitations.

The repository provides the scripts, saved split, checkpoints, and experimental outputs required to reproduce the reported results.