# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Function to generate base models without resampling
def generateBaseModels(X_train, Y_train, models):
    baseModels = []
    for i, (name, model) in enumerate(models.items()):
        # Fit each model on the full training set
        model.fit(X_train, Y_train)
        
        # Append the fitted model to the list of base models
        baseModels.append(model)
    return baseModels

# Function to generate the meta-model based on the base models' predictions
def generateMetaModel(X_train, Y_train, models):
    
    # Split the training data into training and validation sets for meta-model training
    X_train_train, X_train_val, Y_train_train, Y_train_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=63, stratify=Y_train)
    
    # Generate base models using the reduced training data
    baseModels = generateBaseModels(X_train_train, Y_train_train, models)
    
    # Generate predictions on the validation set from each base model
    y_pred = []

    for model in baseModels:
        y_pred.append(model.predict(X_train_val))
    
    # Stack the predictions for meta-model input
    y_pred = np.column_stack(y_pred)
    
    # Define the meta-model
    metaModel = LogisticRegression(random_state=63)
    
    # Train the meta-model on the stacked base model predictions
    metaModel.fit(y_pred, Y_train_val)
    
    return baseModels, metaModel

# Function to generate ensemble predictions based on stacking or voting methods
def ensemblePrediction(baseModels, metaModel, X_test, method="stacking"):
    
    # Generate predictions from each base model on the test data
    y_pred_base = []
    
    for model in baseModels:
        y_pred_base.append(model.predict(X_test))
    
    # Stack predictions for ensemble
    y_pred_base = np.column_stack(y_pred_base)
    
    # For stacking, use the meta-model to make the final prediction
    if method == "stacking":
        return metaModel.predict(y_pred_base)
    
    # For voting, take the average prediction and threshold at 0.5
    elif method == "voting":
        y_pred_avg = np.mean(y_pred_base, axis=1)
        return (y_pred_avg > 0.5).astype(int)

# Function to calculate performance metrics
def calculatePerformanceMetrics(y_true, y_pred):
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Sensitivity (Recall) and Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Calculate Precision
    precision = precision_score(y_true, y_pred)
    
    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # Calculate Area Under the Receiver Operating Characteristic Curve (AUROC)
    auroc = roc_auc_score(y_true, y_pred)
    
    # Calculate Area Under Precision-Recall Curve (AUPR)
    aupr = average_precision_score(y_true, y_pred)
    
    # Return all metrics in a dictionary
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auroc': auroc,
        'aupr': aupr
    }

# Load and preprocess the dataset
file_path = 'water_potability.csv'
data = pd.read_csv(file_path)

# Drop duplicates and missing values in the target column
data.drop_duplicates(inplace=True)
data.dropna(subset=['Potability'], inplace=True)

# Separate features and target variable
X = data.drop(columns="Potability")
y = data["Potability"]

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Standardize the features for uniform scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=83)

# Define various models for the ensemble
models = {
    "Logistic Regression": LogisticRegression(random_state=63),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Support Vector Machine": SVC(kernel='rbf', random_state=63),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=63),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=63),
    "XGBoost": XGBClassifier(n_estimators=1000, random_state=63, use_label_encoder=False, eval_metric='logloss', learning_rate=0.01),
    "Decision Tree": DecisionTreeClassifier(max_depth=9, random_state=63),
    "Tabnet": TabNetClassifier(optimizer_params=dict(lr=2e-2))
}

# Generate base models
baseModels = generateBaseModels(X_train, Y_train, models)

# Generate base models and meta-model for stacking ensemble
baseModelsStack, metaModel = generateMetaModel(X_train, Y_train, models)

# Predictions for Voting and Stacking Ensembles
y_pred_vote = ensemblePrediction(baseModels, metaModel, X_test, method="voting")
y_pred_stack = ensemblePrediction(baseModelsStack, metaModel, X_test, method="stacking")

# Calculate and print performance metrics for Voting Ensemble
performanceMetrics_vote = calculatePerformanceMetrics(Y_test, y_pred_vote)
performanceMetrics_vote_df = pd.DataFrame(performanceMetrics_vote, index=[0])
print("Voting Ensemble Metrics:\n", performanceMetrics_vote_df)

# Calculate and print performance metrics for Stacking Ensemble
performanceMetrics_stack = calculatePerformanceMetrics(Y_test, y_pred_stack)
performanceMetrics_stack_df = pd.DataFrame(performanceMetrics_stack, index=[0])
print("Stacking Ensemble Metrics:\n", performanceMetrics_stack_df)
