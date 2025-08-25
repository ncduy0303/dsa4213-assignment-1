# DSA4213 Assignment 1: Word Embeddings

This repository contains the implementation and analysis for Assignment 1 of the course DSA4213: Natural Language Processing for Data Science. The objective of this assignment is to understand, implement, and compare key word embedding algorithms.

## 📝 About The Project

This project focuses on three word embedding techniques:

1. **Skip-gram** with negative sampling
2. **SPPMI-SVD** (Shifted Positive Pointwise Mutual Information with Singular Value Decomposition)
3. **GloVe** (Global Vectors for Word Representation)

These models are trained on the **Corpus of Singapore English Messages (CoSEM)**. The trained embeddings are then evaluated and compared qualitatively by examining nearest neighbors and visualizing the embedding space using dimensionality reduction techniques like PCA or t-SNE.

The entire workflow, from data preprocessing to model training and evaluation, is documented in the `Assignment1.ipynb` Jupyter Notebook.

## 🚀 Getting Started

1. Clone this repository to your local machine.
2. Open the `Assignment1.ipynb` notebook in Jupyter.
3. Run the cells sequentially.

**Note on Data and Models:**
The notebook is designed for reproducibility.

- Upon first run, it will automatically create a `data/` directory and download the CoSEM corpus.
- It will also create a `weights/` directory to store the trained model embeddings.
- If you re-run the notebook, it will detect the existing data and model files and will not re-download or re-train them, saving time. To force a re-training, simply delete the contents of the `weights/` folder.

## 📋 Assignment Objectives

- Understand and implement key word embedding algorithms.
- Train embeddings using a real corpus.
- Compare model outputs via qualitative analyses.
- Practice scientific reporting and reproducible research.
