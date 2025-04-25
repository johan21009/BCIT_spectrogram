# BCIT_spectrogram

This repository contains the complete pipeline and scripts used in the research project titled:

**"EEG-Based Prediction of Driving Performance Using Spectrograms and Vision Transformers"**

## 🧠 Project Overview

This project aims to predict driver performance in a simulated driving environment using EEG data and advanced deep learning models. The approach focuses on transforming EEG signals into spectrogram representations and classifying them using a Vision Transformer (ViT) model.

The methodology integrates several signal processing techniques including:
- EEG artifact removal via Independent Component Analysis (ICA)
- Extraction of Event-Related (De)Synchronization (ERDS) patterns
- Spectrogram generation
- Vision Transformer model training for classification (Good vs. Bad performance)

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `extract_raw_performance_labels.ipynb` | Script for extracting reaction time-based performance labels from the driving simulation. |
| `extract_spectrogram.ipynb` | Jupyter notebook to generate spectrograms from preprocessed EEG data. |
| `script_extract_spectrogram.py` | Python script version of the spectrogram extraction pipeline. |
| `script_extract_spectrogram_CH-ERDS.py` | Extracts channel-wise ERDS-based spectrograms for each EEG segment. |
| `transformers_performance.ipynb` | Trains a Vision Transformer (ViT) on spectrogram images to classify driving performance. |
| `transformers_performance-EEG.ipynb` | Alternate or extended version of the ViT training notebook. |
| `paths_with_labels.csv` | Contains paths to spectrogram images with associated Good/Bad performance labels. |
| `README.md` | This file. |
| `.gitignore` | Git ignore rules. |

## 📊 Dataset

The EEG dataset used in this project is from the **BCIT Mind Wandering Study**, which includes data from 21 participants performing a 30-minute simulated driving task under various audio conditions. 

Each subject was exposed to lateral perturbations while driving, and EEG data were collected to measure reaction times and attention levels.

- Dataset Source: [OpenNeuro - BCIT Mind Wandering](https://doi.org/10.18112/openneuro.ds004121.v1.0.0)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BCIT_spectrogram.git
   cd BCIT_spectrogram
    ```
2. Set up your Python environment:

- Python 3.8+
- Required libraries: numpy, pandas, mne, matplotlib, opencv-python, torch, timm, scikit-learn, Pillow

3. Extract performance labels:
   ```bash
   python extract_raw_performance_labels.ipynb
   ```

4. Generate spectrograms (choose one of the scripts based on your approach):
   ```bash
   python script_extract_spectrogram.py
   ```

5. Train the transformer model: Open transformers_performance.ipynb and run the training pipeline.

## 📈 Results
Using the proposed method with ERDS-enhanced spectrograms and Vision Transformers, the model achieved:

- Global Accuracy: 81.8%
- ROC-AUC: 89.8%
- F1 Score (macro): 82.1%

These results demonstrate a strong ability to generalize across subjects and distinguish between good and poor driving performance using EEG.

## 🧪 Citation
If you use this repository or dataset, please cite the dataset source:

Touryan, J., Apker, G. (2022). BCIT Mind Wandering. OpenNeuro. https://doi.org/10.18112/openneuro.ds004121.v1.0.0


