# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier

# Load and preprocess the dataset
data = pd.read_csv('water_potability.csv')
data.drop_duplicates(inplace=True)
data.dropna(subset=['Potability'], inplace=True)

# Feature selection based on correlation with the target
#correlation = data.corr()["Potability"].abs().sort_values(ascending=False)
#features = correlation[1:6].index.tolist()
#data = data[features + ["Potability"]]

print(data.head())

# Separate features and target variable
X = data.drop(columns="Potability")
y = data["Potability"]

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=83)

# Define Models
xgb_model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
tabnet_model = TabNetClassifier(optimizer_params=dict(lr=2e-2))

# Train Models
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
tabnet_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], patience=10, max_epochs=200, batch_size=128)

# Generate Predictions
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict_proba(X_test)[:, 1]
tabnet_preds = tabnet_model.predict_proba(X_test)[:, 1]

# Ensemble Predictions by Averaging
ensemble_preds = (xgb_preds + rf_preds + tabnet_preds) / 3
final_preds = (ensemble_preds > 0.5).astype(int)

# Evaluation
print("Ensemble Model Accuracy:", accuracy_score(y_test, final_preds))
print("\nClassification Report:\n", classification_report(y_test, final_preds))
