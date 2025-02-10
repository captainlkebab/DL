# Spotify Genre Classification

## Overview
This project focuses on building a machine learning model using PyTorch to classify Spotify songs into different genres based on various audio features. The dataset consists of 30,000 Spotify songs with features such as danceability, energy, loudness, and tempo.

## Features and Approach
- **Feature Selection**: Identified the most relevant features using ANOVA tests.
- **Feature Engineering**: Converted categorical features into numerical labels.
- **Data Preprocessing**: Handled missing values, normalized numerical features, and split the dataset into training and validation sets.
- **Model Architecture**: Implemented a Multi-Layer Perceptron (MLP) with two hidden layers, batch normalization, dropout, and ReLU activations.
- **Training**: Used CrossEntropyLoss, Stochastic Gradient Descent (SGD) with momentum, learning rate scheduling, and early stopping.
- **Evaluation**: Assessed model performance using accuracy and loss curves.

## Installation and Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib
```

## Running the Code
1. Download the dataset:
   ```bash
   kaggle datasets download joebeachcapital/30000-spotify-songs
   unzip 30000-spotify-songs.zip
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook spotify_dl.ipynb
   ```
3. Execute the cells in sequence to train and evaluate the model.

## Results
- The model achieves 55-60% accuracy in genre classification, quite low, even with adjustment to Learning Rate, optimizer, Layers...s
- Loss curves are plotted to track training progress.
- Further improvements can be made by experimenting with RNN-based models for sequential learning.

## Future Improvements
- Implement an RNN/LSTM version for potential sequential patterns in music data.
- Tune hyperparameters further (e.g., learning rate, batch size).
- Use additional evaluation metrics such as precision, recall, and F1-score.

---
Author: Samil
