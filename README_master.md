# Welcome to the Lemanic Life Science Hackathon : Project 4 - LostLoss in Translation

### AI for Modeling Gene Translation from RNA to Proteins in Immune Cells

## Overview

*LostLoss in Translation* is an interdisciplinary machine learning project that aims to model gene translation in single cells by predicting **protein abundance from RNA expression**.

Understanding how RNA expression translates into protein levels is a key challenge in computational biology, with direct applications in:

* Cell type annotation
* Understanding immune responses in disease
* Improving biological interpretability of single-cell data

We leverage a large-scale, real-world **CITE-seq dataset** curated by CHUV, containing paired RNA and protein measurements at single-cell resolution.

---

## Project Goals

This project focuses on building and benchmarking predictive models that infer protein abundance from RNA data using two complementary approaches:

### 1. Classical & Deep Learning Models

* Supervised learning pipelines
* Baseline regression models (e.g. linear regression, random forests, LightGBM)
* Deep neural networks (e.g. MLPs, PyTorch-based architectures)

### 2. Foundation Model-Based Approaches

* Leveraging RNA and biological language foundation models
* Using embeddings from pretrained models (e.g. HuggingFace ecosystem)
* Exploring transfer learning for cross-modal prediction

---

## Dataset

We use a curated **CITE-seq dataset** provided by CHUV, which includes:

* Single-cell RNA expression profiles
* Matching protein abundance measurements (ADT data)
* Associated metadata for cell annotation and experimental context

### Format

* Data is provided in **`.h5ad` format (AnnData)**
* Includes preprocessed and analysis-ready matrices
* Accompanied by a Jupyter notebook tutorial for loading, exploring, and visualizing the dataset in Python

---

## Scientific Skills & Requirements

### Programming & Data Science

* Python (NumPy, pandas, matplotlib, seaborn)
* Unix/Linux command line
* Git & version control
* Conda environment management

### Machine Learning

* Core ML concepts:

  * Feature selection
  * Dimensionality reduction
  * Train/test splitting
  * Cross-validation
* Classical ML models:

  * Regression, SVM, LightGBM (scikit-learn ecosystem)
* Deep learning:

  * PyTorch or PyTorch Lightning (preferred)
  * Model training and evaluation workflows

### Advanced / Optional (Highly Valuable)

* Experience with foundation models or LLMs (Hugging Face ecosystem)
* GPU computing / HPC environments
* Computational biology / bioinformatics background
* Single-cell omics (transcriptomics, proteomics, CITE-seq)

---

## Data Availability

The dataset will be distributed at the start of the hackathon
* Preprocessed and ready-to-use
* Delivered in `.h5ad` format
* Accompanied by a a dataset description
---

## Institution

Developed in collaboration with **CHUV (Centre Hospitalier Universitaire Vaudois)**, **EPFL-ISREC**.

