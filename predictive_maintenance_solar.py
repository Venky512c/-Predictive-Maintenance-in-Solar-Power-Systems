
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate dataset
np.random.seed(42)
data_size = 1000

df = pd.DataFrame({
    'temperature': np.random.normal(45, 5, data_size),
    'voltage': np.random.normal(600, 50, data_size),
    'current': np.random.normal(5, 1, data_size),
    'maintenance_count': np.random.randint(0, 10, data_size),
    'operating_hours': np.random.normal(2000, 300, data_size),
    'ambient_temp': np.random.normal(30, 3, data_size),
    'failure': np.random.choice([0, 1], size=data_size, p=[0.9, 0.1])  # imbalanced data
})

# Preprocessing
X = df.drop(columns=['failure'])
y = df['failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance visualization
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Predicting Inverter Failure")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
