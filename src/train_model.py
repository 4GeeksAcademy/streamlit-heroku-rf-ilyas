import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Iris data
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split (optional)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create models dir path
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'iris_model.pkl')

# Save model there
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved at {model_path}")