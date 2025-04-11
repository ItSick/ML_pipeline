import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

class ClassifierModel:
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probabilities."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability predictions.")

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation and return scores."""
        scores = cross_val_score(self.model, X, y, cv=cv)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        return scores

    def evaluate(self, X, y):
        """Evaluate the model using classification report."""
        y_pred = self.predict(X)
        print(classification_report(y, y_pred))
        return classification_report(y, y_pred, output_dict=True)

    def save_model(self, filepath):
        """Save the model to a file."""
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
