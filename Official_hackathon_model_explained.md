# Hackathon Baseline Model — Explained

This document explains how `baseline_model_hackathon.ipynb` works, step by step.

---

## What is this notebook doing?

The goal is simple: **given a cell's RNA data, predict which proteins that cell expresses.**

More specifically:
- **Input**: RNA gene expression for a single cell (33,694 genes, compressed down to 1,000 numbers)
- **Output**: For each of 273 proteins, predict: is this protein **expressed (1)** or **not expressed (0)**?

This is a **binary classification** task. For every cell, the model makes 273 yes/no predictions: one per protein.

---

## Where does this come from?

This notebook is adapted from a **2nd-place Kaggle competition solution**. The journey was:

1. **Kaggle original** (`2nd-place-gru-cite.ipynb`) — regression model predicting continuous protein expression from RNA. Used 140 proteins and a different dataset.
2. **Hackathon regression version** (`hackathon_model.ipynb`) — same model architecture, adapted to work on the GSE194315 hackathon dataset (273 proteins, 180k cells, loaded from `.h5mu` format).
3. **Hackathon classification version** (`hackathon_model_updated.ipynb`) — the model architectures stay the same, but the task is changed from regression (predict a continuous number) to classification (predict 0 or 1). This is the notebook explained here.

The reason for switching to classification: we will compare this baseline to a contrastive learning (CLIP-style) model, and it's much easier to compare models when they both predict binary outcomes rather than continuous values.

---

## Step-by-step walkthrough

### 1. Loading the data

The notebook loads a MuData file (`GSE194315_raw_mu.h5mu`) which contains two modalities measured from the same cells using CITE-seq:

- **RNA** (gene expression): 180,794 cells x 33,694 genes — stored as a sparse matrix of raw integer counts
- **Protein** (ADT / antibody-derived tags): 180,794 cells x 273 proteins — also raw integer counts

Think of it this way: each row is one cell. The RNA columns tell you which genes are active in that cell. The protein columns tell you which proteins are on the cell's surface.

**Subsampling**: For development speed, the notebook randomly picks 30,000 cells instead of using all 180k. Set `SUBSAMPLE_N = None` to use the full dataset.

**Dead cell removal**: Some cells have identical protein values across all 273 proteins (standard deviation = 0). These are likely dead or damaged cells. They get removed (~7,800 cells removed from the 30k subsample).

### 2. Binarising the targets

This is the key change from the regression version. Instead of trying to predict the exact protein count, we simplify to just: **is this protein expressed or not?**

How it works (won't be needed for already normalised data):
1. For each of the 273 proteins, compute the **median** expression value across all cells
2. If a cell's expression for a protein is **above that protein's median** → label it **1** (expressed)
3. If it's **at or below the median** → label it **0** (not expressed)

Each protein gets its own threshold because different proteins have very different baseline expression levels. For example, CD4 might have a median of 50 while a rare protein might have a median of 0.

### 3. RNA preprocessing

**Step 1 — Library size normalisation**

**Step 2 — Log1p transform**

**Step 3 — TruncatedSVD (dimensionality reduction)**: Avoid computational expense

### 4. The DataGenerator

Neural networks train on batches (groups of cells processed together). The `DataGenerator` class handles this:
- It takes the full dataset and serves it in chunks of `BATCH_SIZE` cells 
- During training, it shuffles the data each epoch so the model sees cells in a different order
- This is more memory-efficient than loading everything at once

### 5. Cross-validation (K-Fold)

Instead of a single train/test split, the notebook uses **5-fold cross-validation**:

1. Split all cells into 5 equal groups
2. Train on 4 groups, validate on the 1 held-out group
3. Repeat 5 times, each time holding out a different group
4. Every cell gets a prediction exactly once (when it was in the held-out group)

This gives a more reliable estimate of model performance. The "out-of-fold" (OOF) predictions are collected for all cells and used to compute the final accuracy.

### 6. Training procedure

For each fold, the training loop does:

1. **Build the model** from scratch
2. **Train** for up to 100 epochs on the training cells
3. **Early stopping**: if validation loss doesn't improve for 10 epochs, stop training since the model has learned enough and would start overfitting
4. **Learning rate reduction**: if validation loss stalls for 6 epochs, multiply the learning rate by a factor (e.g., 0.05 or 0.1) to take smaller steps
5. **Checkpoint**: save the best model weights (lowest validation loss)
6. **Predict**: load the best weights and predict on the held-out validation cells
7. **Evaluate**: compute accuracy on this fold

### 7. Interpreting the model's output

The model outputs a tensor of shape `(batch_size, 273, 2)` : for each cell, for each of the 273 proteins, it outputs **2 raw scores**:

- Score[0]: confidence that this protein is NOT expressed
- Score[1]: confidence that this protein IS expressed

These raw scores are converted to probabilities using **softmax** (makes them sum to 1). Then we take the probability of class 1 (expressed). If that probability is >= 0.5, we predict "expressed", otherwise "not expressed".

---

## The two model architectures

Both models take the same input (1,000 SVD features) and produce the same output (273 x 2 logits). They differ in their internal structure.

### Model 1 — GRU-first model

```
Input (1000 features)
  |
  v
Reshape to (1, 1000)          -- pretend the features are a sequence of length 1
  |
  v
Bidirectional GRU (1800 units, elu activation, identity init)
  |                            -- the GRU processes the input and outputs 3600 features
  v                               (1800 forward + 1800 backward)
GaussianDropout (0.2)  -----> [x1]
  |
  v
Dense (1800, elu, identity init)
  |
  v
GaussianDropout (0.2)  -----> [x3]
  |
  v
Dense (1800, elu, identity init)
  |
  v
GaussianDropout (0.2)  -----> [x5]
  |
  v
Concatenate [x1, x3, x5]      -- combine all three stages: 3600 + 1800 + 1800 = 7200 features
  |
  v
Dense (273 * 2 = 546)         -- linear projection to 546 logits
  |
  v
Reshape to (273, 2)           -- 2 scores per protein
```


### Model 2 — Dense-first model

```
Input (1000 features)
  |
  v
Dense (1500, swish)
  |
  v
GaussianDropout (0.1)
  |
  v
Dense (1500, swish)
  |
  v
GaussianDropout (0.1)
  |
  v
Dense (1500, swish)
  |
  v
GaussianDropout (0.1)
  |
  v
Reshape to (1, 1500)
  |
  v
Bidirectional GRU (700 units, swish activation)
  |                            -- outputs 1400 features (700 forward + 700 backward)
  v
GaussianDropout (0.1)
  |
  v
Dense (273 * 2 = 546)         -- linear projection to 546 logits
  |
  v
Reshape to (273, 2)           -- 2 scores per protein
```

### Why two models?

Different architectures learn different patterns. By training both and averaging their predictions, we get a more robust result. This is called **ensembling** or **blending**.

---

## The blending step

After both models produce predictions (probabilities of "expressed" for each protein), we average them:

```
final_prediction = model1_prediction * 0.55 + model2_prediction * 0.45
```

This weighted average tends to outperform either model alone because the two architectures have different advantages. Model 1 gets slightly more weight (0.55) than Model 2 (0.45)(was done like this in the original competition notebook).

The final accuracy is computed on these blended predictions.

---

## Loss function: Sparse Categorical Cross-Entropy

Both models use `SparseCategoricalCrossentropy(from_logits=True)`.

What this means:
- **Categorical**: we're classifying into categories (0 = not expressed, 1 = expressed)
- **Sparse**: the targets are integer labels (0 or 1), not one-hot encoded vectors ([1,0] or [0,1])
- **Cross-entropy**: the standard loss for classification

---

## Evaluation metric: Accuracy

The notebook reports a single metric: **accuracy**.

With the subsampled dataset (30k cells), the baseline achieves roughly **77.5% accuracy**.

---

## Summary of what is different from the original Kaggle model

| Aspect | Kaggle original | This notebook |
|--------|----------------|---------------|
| Dataset | Kaggle competition data | GSE194315 (hackathon) |
| Data format | Pre-processed numpy arrays | Raw .h5mu file |
| Proteins | 140 | 273 |
| Cells | 70,988 | 180,794 (subsampled to 30k for dev) |
| Task | Regression (predict continuous values) | Classification (predict 0 or 1) |
| Targets | Raw continuous protein expression | Binarised |
| Output layer | Dense(140, linear) | Dense(546, linear) + Reshape(273, 2) |
| Loss | Cosine similarity / MSE | Sparse Categorical Cross-Entropy |
| Evaluation | Pearson correlation | Accuracy |
| Model internals | Identical | Identical |
