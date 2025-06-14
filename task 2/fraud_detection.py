import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Create synthetic fraud detection data
np.random.seed(42)
n = 500
data = pd.DataFrame({
    "amount": np.random.uniform(10, 1000, n),
    "oldbalanceOrg": np.random.uniform(0, 2000, n),
    "newbalanceOrig": np.random.uniform(0, 2000, n),
    "oldbalanceDest": np.random.uniform(0, 3000, n),
    "newbalanceDest": np.random.uniform(0, 3000, n),
    "is_fraud": np.random.choice([0, 1], size=n, p=[0.9, 0.1])
})

# Insert some NaN values
for col in data.columns[:-1]:
    data.loc[data.sample(frac=0.05).index, col] = np.nan

# Step 2: Split features and label
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Step 3: Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))