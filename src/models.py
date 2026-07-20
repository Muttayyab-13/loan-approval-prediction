"""
models.py — define and train the four classification models.

We use simple default settings (no GridSearch, no tuning) to keep it beginner-friendly.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def build_models():
    """Return a dictionary of {model name: fresh model} with simple settings."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }
    return models


def train_models(models, X_train, y_train):
    """Fit every model on the training data. Returns the same dictionary, trained."""
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[models] Trained: {name}")
    return models
