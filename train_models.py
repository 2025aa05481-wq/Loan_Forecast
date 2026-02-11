import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the wine quality dataset"""
    print("Loading wine quality dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        df = pd.read_csv(url, sep=';')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except:
        print("Failed to load from URL, creating synthetic dataset...")
        np.random.seed(42)
        n_samples = 1000
        data = {
            'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
            'volatile_acidity': np.random.normal(0.5, 0.2, n_samples),
            'citric_acid': np.random.normal(0.27, 0.2, n_samples),
            'residual_sugar': np.random.normal(2.5, 1.5, n_samples),
            'chlorides': np.random.normal(0.087, 0.05, n_samples),
            'free_sulfur_dioxide': np.random.normal(15.9, 10.5, n_samples),
            'total_sulfur_dioxide': np.random.normal(46.5, 32.9, n_samples),
            'density': np.random.normal(0.997, 0.002, n_samples),
            'pH': np.random.normal(3.31, 0.15, n_samples),
            'sulphates': np.random.normal(0.65, 0.17, n_samples),
            'alcohol': np.random.normal(10.4, 1.1, n_samples),
            'quality': np.random.randint(3, 9, n_samples)
        }
        df = pd.DataFrame(data)
    
    # Convert quality to binary classification
    df['quality_category'] = np.where(df['quality'] >= 6, 'good', 'bad')
    
    # Prepare features and target
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Calculate AUC score
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
    else:
        auc = 0.5
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }

def main():
    """Main function to train and save all models"""
    print("=" * 60)
    print("TRAINING ML CLASSIFICATION MODELS")
    print("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_prepare_data()
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
    
    # Save models and preprocessing objects
    print("\nSaving models and preprocessing objects...")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save trained models
    for name, model in trained_models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} as {filename}")
    
    # Save results
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Display comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']:<10.4f} {metrics['AUC']:<10.4f} "
              f"{metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} "
              f"{metrics['F1 Score']:<10.4f} {metrics['MCC']:<10.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()
