import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    print("Loading dataset into memory...")
    try:
        df = pd.read_csv('isl_features.csv')
    except FileNotFoundError:
        print("Error: isl_features.csv not found. Run extract.py first.")
        return

    # Verify what we extracted
    num_classes = df['label'].nunique()
    print(f"Data loaded! Found {len(df)} total images across {num_classes} classes.")

    X = df.drop('label', axis=1)
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Splitting data into 80% training and 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # n_jobs=-1 utilizes all CPU cores for significantly faster training
    print("Training Random Forest Classifier (this may take a few seconds)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 50)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    
    # Optional: Print detailed accuracy for each individual letter
    print("Generating detailed classification report...")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/rf_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Success: rf_model.pkl and label_encoder.pkl updated and saved.")

if __name__ == "__main__":
    train_model()