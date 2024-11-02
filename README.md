# Water Potability Prediction

This repository contains a comprehensive machine learning project focused on predicting water potability based on a set of physicochemical parameters. The project encompasses data preprocessing, model training, ensemble learning, and performance evaluation to accurately classify water samples as potable or non-potable. The primary objective is to leverage machine learning techniques to assess water quality for safe consumption.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Ensemble Learning](#ensemble-learning)
- [Performance Metric](#performance-metric)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)

## Project Overview

The goal of this project is to classify water samples as either potable or non-potable by analyzing their chemical properties. Using a dataset of water samples with attributes like pH, hardness, and solids, this project trains and compares various machine learning models. Advanced ensemble learning techniques, such as stacking and voting, are also applied to enhance the predictive accuracy. The project is designed to showcase the effectiveness of combining individual models to improve overall performance.

## Dataset

The dataset used in this project is sourced from Kaggle's **Water Potability Dataset** and contains the following columns:
- **pH**: pH value of the water sample.
- **Hardness**: Measure of water hardness, which affects scaling in pipelines.
- **Solids**: Total dissolved solids in parts per million (ppm).
- **Chloramines**: Chloramines concentration in ppm, an indicator of water disinfection.
- **Sulfate**: Sulfate concentration in ppm, which may impact taste and odor.
- **Conductivity**: Electrical conductivity in μS/cm, indicating the ability to conduct electricity.
- **Organic Carbon**: Organic carbon concentration in ppm, associated with natural organic matter in water.
- **Trihalomethanes**: Concentration in ppm, a byproduct of chlorination and a potential carcinogen.
- **Turbidity**: Turbidity level in NTU, affecting water clarity.
- **Potability**: Target variable indicating water potability (1: Potable, 0: Not Potable).

## Features

### Data Preprocessing
- **Handling Missing Values**: Missing values are imputed with the mean to ensure data completeness.
- **Scaling**: Standardization is applied to bring features to a similar scale, aiding model performance.
  
### Feature Engineering
- **Correlation Analysis**: The top features correlated with potability are selected to enhance model input quality.

### Model Selection
A range of supervised learning algorithms is used, including:
- **Logistic Regression**: A basic model for binary classification.
- **K-Nearest Neighbors (KNN)**: A distance-based classifier.
- **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane for classification.
- **Random Forest**: An ensemble of decision trees to enhance accuracy.
- **AdaBoost**: An adaptive boosting technique that emphasizes difficult samples.
- **XGBoost**: A gradient boosting model optimized for performance.
- **Decision Tree**: A simple tree-based model for interpretability.

## Modeling Approach

### Baseline Models
Each individual model is trained on the dataset, and key performance metrics are recorded to establish baseline accuracies.

### Ensemble Models
Two main ensemble approaches are used to combine the strengths of individual models:
- **Stacking Ensemble**: Combines predictions of base models using a Logistic Regression meta-model.
- **Voting Ensemble**: Uses a voting system across models to decide the final class prediction.
- **Comparison**: Visualizations and performance metrics are provided to compare individual models with ensemble models.

## Ensemble Learning

### Stacking
In stacking, predictions from each base model are used as inputs for a meta-model, which learns to make the final prediction based on the strengths of the individual models. This method enhances accuracy by allowing the meta-model to correct errors from base models.

### Voting
In the voting ensemble, predictions from each model are averaged, and the sample is classified based on a majority threshold, which improves robustness and reduces overfitting.

## Performance Metric

The primary metric used to evaluate model performance in this project is:
- **Accuracy**: Measures the percentage of correct predictions, providing a straightforward evaluation of model effectiveness on the water potability classification task.

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

## Usage

1. **Visualization**:
   - View `Model_Comparison.png` for a bar chart visualization of model accuracies, highlighting Support Vector Machine as the top-performing model with an accuracy of 0.72.

2. **Results Summary**:
   - The **Results.txt** file contains detailed accuracy scores for each model, including ensemble approaches. Key results:
     - Logistic Regression: 0.64
     - Decision Tree: 0.66
     - Random Forest: 0.68
     - K-Nearest Neighbors: 0.67
     - Support Vector Machine: 0.72
     - AdaBoost: 0.64
     - XGBoost: 0.68
     - Ensemble Model (XGB-RF-TabNet): 0.70
     - Voting Ensemble Model: 0.69
     - Stacking Ensemble Model: 0.71

## Results

The project results show that the Support Vector Machine model achieved the highest accuracy among individual models (0.72), followed closely by Random Forest and XGBoost. The ensemble models, particularly the stacking ensemble, demonstrated improved performance, combining the strengths of individual models to yield a more robust classifier.

The **Results.txt** file provides a detailed breakdown of individual model and ensemble accuracies, while **Model_Comparison.png** visually compares the models, illustrating the differences in performance. These results emphasize the effectiveness of ensemble learning for water potability prediction.

## Contributors

- **Azim Nahin**  
- **Sadman1702042**

---
