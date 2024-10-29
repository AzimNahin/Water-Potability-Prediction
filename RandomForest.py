import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('water_potability.csv')

# Drop duplicate values
data.drop_duplicates(inplace=True)

# Drop rows with missing target values
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=83)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=83)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
print(f"Accuracy of Random Forest model: {accuracy:.2f}")
