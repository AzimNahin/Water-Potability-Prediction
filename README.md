# Water Potability Prediction

This repository contains a machine learning project focused on predicting water potability based on various physicochemical parameters. The project involves data preprocessing, model selection, ensemble learning techniques, and performance evaluation for accurate prediction of water safety.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Ensemble Learning](#ensemble-learning)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)

## Project Overview
The aim of this project is to classify water samples as potable or non-potable based on various chemical properties. It uses a dataset with attributes like pH, hardness, and solids to train a variety of machine learning models and evaluate their performance.

## Dataset
The dataset used for this project is sourced from [Kaggle's Water Potability Dataset](https://www.kaggle.com/adityakadiwal/water-potability) and contains the following columns:
- **pH**: pH value of water
- **Hardness**: Measures water hardness
- **Solids**: Total dissolved solids (ppm)
- **Chloramines**: Chloramines level in ppm
- **Sulfate**: Sulfate concentration in ppm
- **Conductivity**: Electrical conductivity in μS/cm
- **Organic Carbon**: Organic carbon concentration in ppm
- **Trihalomethanes**: Trihalomethanes concentration in ppm
- **Turbidity**: Turbidity level in NTU
- **Potability**: Indicates water potability (1: Potable, 0: Not Potable)

## Features
- **Data Preprocessing**: Includes handling missing values, scaling, and imputing with mean values.
- **Feature Engineering**: Top features based on correlation are selected for model input.
- **Model Selection**: Models used include Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, AdaBoost, XGBoost, and Decision Tree.

## Modeling Approach
1. **Baseline Models**: Initial models trained individually to evaluate their performance.
2. **Stacking Ensemble**: Combines predictions of base models using a Logistic Regression meta-model.
3. **Voting Ensemble**: Uses a voting system across models to decide the final class prediction.

## Ensemble Learning
The stacking and voting ensemble methods are used to combine the strengths of multiple models, improving accuracy and robustness.

### Stacking
In stacking, predictions from each base model are combined, and a meta-model is trained on these predictions to make the final prediction.

### Voting
In voting, predictions from each model are averaged, and the final prediction is based on a threshold, classifying the sample as potable or non-potable.

## Performance Metrics
The following metrics are used to evaluate model performance:
- **Accuracy**: Overall accuracy of the predictions
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **F1 Score**: Balance between precision and sensitivity
- **AUROC**: Area Under the Receiver Operating Characteristic Curve
- **AUPR**: Area Under the Precision-Recall Curve

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AzimNahin/Water-Potability-Prediction.git
2. Navigate to the project directory:
   ```bash
   cd Water-Potability-Prediction
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

## Results
Results from the model evaluation are displayed in terms of accuracy, sensitivity, specificity, and other metrics. Ensemble models, particularly the stacking ensemble, generally perform well for this dataset.

## Contributors
- [Azim Nahin](https://github.com/AzimNahin)
- [Sadman1702042](https://github.com/Sadman1702042)

