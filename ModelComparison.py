import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
file_path = 'water_potability.csv'
data = pd.read_csv(file_path)
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=83)

# Define the models to be compared
models = {
    "Logistic Regression": LogisticRegression(random_state=63),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Support Vector Machine": SVC(kernel='rbf', random_state=63),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=63),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=63),
    "XGBoost": XGBClassifier(n_estimators=1000, random_state=63, use_label_encoder=False, eval_metric='logloss', learning_rate=0.01),
    "Decision Tree": DecisionTreeClassifier(max_depth=9, random_state=63)
}

# Train each model, predict, and calculate accuracy
accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[model_name] = accuracy_score(y_test, y_pred)

# Enhanced visualization of accuracies
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
bar_plot = sns.barplot(
    x=list(sorted_accuracies.values()), 
    y=list(sorted_accuracies.keys()), 
    palette="plasma", 
    dodge=False, 
    edgecolor="black"
)

# Adding annotations for each accuracy value
for index, (model, accuracy) in enumerate(sorted_accuracies.items()):
    bar_plot.text(
        accuracy - 0.05, index, 
        f"{accuracy:.2f}", 
        color='black', 
        ha="center", 
        va="center", 
        fontweight="bold", 
        fontsize=10, 
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
    )

# Set plot labels and title
plt.xlabel("Accuracy", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Model", fontsize=12, fontweight="bold", labelpad=10)
plt.title("Model Accuracies on Water Potability Dataset", fontsize=14, fontweight="bold", pad=15)
plt.xlim(0, 1)

# Customize plot layout
plt.xticks(fontsize=10, fontweight="medium")
plt.yticks(fontsize=10, fontweight="medium")
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout(pad=1.5)
plt.show()
