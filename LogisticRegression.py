import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('water_potability.csv')

# Drop duplicates values
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

# Fill missing values with the mean
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3032)

# Initialize logistic regression model
LR = LogisticRegression(max_iter=1000)

# Train the model
LR.fit(X_train, y_train)

# Make predictions 
y_pred = LR.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
print(f"Accuracy of Logistic Regression model: {accuracy:.2f}")
