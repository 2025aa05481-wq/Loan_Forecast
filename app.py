import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Machine Learning Classification Models Comparison")
st.markdown("""
This application demonstrates the performance comparison of 6 different classification models:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes Classifier
- Random Forest (Ensemble)
- XGBoost (Ensemble)
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Model Comparison", "Dataset Upload", "Model Prediction"])

# Load sample dataset (Wine Quality)
@st.cache_data
def load_sample_dataset():
    """Load the Wine Quality dataset as sample data"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(url, sep=';')
        # Convert quality to binary classification (good vs bad)
        df['quality_category'] = np.where(df['quality'] >= 6, 'good', 'bad')
        return df
    except:
        # Fallback to create synthetic dataset if URL fails
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
        df['quality_category'] = np.where(df['quality'] >= 6, 'good', 'bad')
        return df

# Initialize models
def initialize_models():
    """Initialize all 6 classification models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    return models

# Train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models"""
    models = initialize_models()
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
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
        
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'MCC': mcc,
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results, trained_models

# Home page
if page == "Home":
    st.header("Welcome to ML Classification Models Comparison")
    
    # Load and display sample dataset
    df = load_sample_dataset()
    
    st.subheader("Sample Dataset: Wine Quality Classification")
    st.write(f"Dataset Shape: {df.shape}")
    st.write("Features:", list(df.columns[:-1]))
    st.write("Target Variable: quality_category (good/bad)")
    
    # Display first few rows
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())
    
    # Display target distribution
    st.subheader("Target Variable Distribution")
    fig = px.pie(df, names='quality_category', title='Distribution of Wine Quality')
    st.plotly_chart(fig, use_container_width=True)

# Model Comparison page
elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    # Load and prepare data
    df = load_sample_dataset()
    
    # Prepare features and target
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate models
    with st.spinner("Training and evaluating models..."):
        results, trained_models = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create comparison table
    st.subheader("Model Performance Comparison")
    
    # Prepare data for comparison table
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'AUC': f"{metrics['AUC']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1': f"{metrics['F1 Score']:.4f}",
            'MCC': f"{metrics['MCC']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model selection for detailed analysis
    st.subheader("Detailed Model Analysis")
    selected_model = st.selectbox("Select a model for detailed analysis:", list(results.keys()))
    
    if selected_model:
        # Display metrics
        metrics = results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            for metric_name, value in metrics.items():
                if metric_name not in ['Confusion Matrix', 'Classification Report']:
                    st.metric(metric_name, f"{value:.4f}")
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = metrics['Confusion Matrix']
            fig = px.imshow(cm, text_auto=True, aspect="auto", 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           title=f"Confusion Matrix - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['Classification Report']).transpose()
        st.dataframe(report_df)

# Dataset Upload page
elif page == "Dataset Upload":
    st.header("Upload Your Dataset")
    
    st.write("Upload a CSV file for classification analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success("Dataset uploaded successfully!")
            st.write(f"Dataset Shape: {df.shape}")
            st.dataframe(df.head())
            
            # Ask user to select target column
            target_column = st.selectbox("Select the target column:", df.columns)
            
            if target_column:
                # Prepare features and target
                X = df.drop(target_column, axis=1)
                y = df[target_column]
                
                # Handle categorical variables
                if X.select_dtypes(include=['object']).any().any():
                    st.warning("Dataset contains categorical features. They will be encoded.")
                    X = pd.get_dummies(X, drop_first=True)
                
                # Encode target if it's categorical
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                else:
                    y_encoded = y
                
                # Split and scale data
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Evaluate models
                if st.button("Evaluate Models"):
                    with st.spinner("Training and evaluating models..."):
                        results, trained_models = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
                    
                    # Display results
                    st.subheader("Model Performance Results")
                    
                    comparison_data = []
                    for model_name, metrics in results.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Accuracy': f"{metrics['Accuracy']:.4f}",
                            'AUC': f"{metrics['AUC']:.4f}",
                            'Precision': f"{metrics['Precision']:.4f}",
                            'Recall': f"{metrics['Recall']:.4f}",
                            'F1': f"{metrics['F1 Score']:.4f}",
                            'MCC': f"{metrics['MCC']:.4f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to proceed.")

# Model Prediction page
elif page == "Model Prediction":
    st.header("Make Predictions")
    
    # Load sample data and train models
    df = load_sample_dataset()
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models
    models = initialize_models()
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    st.subheader("Input Features for Prediction")
    
    # Create input fields for each feature
    input_data = {}
    for feature in X.columns:
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            mean_val = df[feature].mean()
            
            input_data[feature] = st.slider(
                feature,
                float(min_val),
                float(max_val),
                float(mean_val),
                step=0.01
            )
    
    # Model selection
    selected_model = st.selectbox("Select model for prediction:", list(trained_models.keys()))
    
    if st.button("Predict"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        model = trained_models[selected_model]
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Display results
        st.subheader("Prediction Results")
        
        # Convert prediction back to original label
        prediction_label = le.inverse_transform([prediction])[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Class", prediction_label)
        
        with col2:
            if prediction_proba is not None:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.2f}%")
        
        # Display probability distribution
        if prediction_proba is not None:
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': le.classes_,
                'Probability': prediction_proba * 100
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability', 
                        title=f"Prediction Probabilities - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This app demonstrates the comparison of 6 different ML classification models on the Wine Quality dataset.

**Models Implemented:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbor
- Naive Bayes
- Random Forest
- XGBoost

**Evaluation Metrics:**
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient
""")
