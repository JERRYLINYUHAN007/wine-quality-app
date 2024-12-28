# Student ID: 11890152
# Student Name: Jerry 林昱瀚

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="White Wine Classification App", layout="wide")

# Function to load and preprocess data
def load_data():
    """
    Load and preprocess the white wine dataset
    Returns: X (features), y (target), raw_df (original dataframe)
    """
    # Load the dataset
    df = pd.read_csv('./data/winequality-white.csv', delimiter=';')
    
    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    return X, y, df

# Function to train models
def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models and return their predictions and scores
    """
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel='rbf', random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    # Train models
    models = {
        'Random Forest': rf_model,
        'Support Vector Machine': svm_model,
        'Neural Network': mlp_model
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'predictions': y_pred
        }
    
    return models, results

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    """
    Create and plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

# Main app
def main():
    """
    Main function to run the Streamlit app
    """
    st.title('White Wine Classification Web App')
    
    # Sidebar
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Set Size', 0.1, 0.4, 0.2)
    
    # Load data
    X, y, df = load_data()
    
    # Data Overview
    st.header('Dataset Overview')
    st.write('Shape of dataset:', df.shape)
    st.write('First few rows of the dataset:')
    st.dataframe(df.head())
    
    # Add statistical summary
    st.subheader('Statistical Summary')
    st.write(df.describe())
    
    # Quality Distribution
    st.subheader('Wine Quality Distribution')
    fig_quality = plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='quality', palette='viridis')
    plt.title('Distribution of Wine Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_quality)
    plt.close()
    
    # Correlation Heatmap
    st.subheader('Feature Correlation Heatmap')
    fig_corr = plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Wine Features')
    st.pyplot(fig_corr)
    plt.close()
    
    # Feature Distributions
    st.subheader('Feature Distributions')
    # Create multiple columns for feature distributions
    cols = st.columns(2)
    features = df.drop('quality', axis=1).columns
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            fig_dist = plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, kde=True)
            plt.title(f'Distribution of {feature}')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_dist)
            plt.close()
    
    # Box Plots for Features vs Quality
    st.subheader('Feature Relationships with Wine Quality')
    for feature in features:
        fig_box = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='quality', y=feature)
        plt.title(f'{feature} vs. Wine Quality')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_box)
        plt.close()
        
    # Add insights about the data
    st.subheader('Key Insights')
    # Quality Distribution Analysis
    quality_stats = df['quality'].value_counts().sort_index()
    most_common_quality = quality_stats.idxmax()
    avg_quality = df['quality'].mean()
    
    st.write(f"""
    1. Quality Distribution:
       - Most common quality score: {most_common_quality}
       - Average quality score: {avg_quality:.2f}
       - Quality scores range from {df['quality'].min()} to {df['quality'].max()}
    
    2. Correlation Analysis:
       - Alcohol shows a {correlation_matrix['quality']['alcohol'] > 0 and 'positive' or 'negative'} correlation with quality (correlation coefficient: {correlation_matrix['quality']['alcohol']:.3f})
       - The strongest correlation with quality is observed for {correlation_matrix['quality'].abs().nlargest(2).index[1]} (coefficient: {correlation_matrix['quality'].abs().nlargest(2).values[1]:.3f})
    
    3. Feature Distributions:
       - Most features show approximately normal distributions
       - Some features have notable outliers, particularly in {', '.join([col for col in df.columns if df[col].skew() > 1])}
    """)
    
    # Data preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models and get results
    models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Display results
    st.header('Model Performance')
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Display metrics in columns
    metrics_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [res['accuracy'] for res in results.values()],
        'Precision': [res['precision'] for res in results.values()],
        'F1 Score': [res['f1'] for res in results.values()]
    })
    
    # Display the metrics table
    st.dataframe(metrics_df)
    
    # Create bar chart for performance metrics
    st.subheader('Model Performance Comparison')
    
    # Reshape data for plotting
    metrics_melted = pd.melt(metrics_df, 
                            id_vars=['Model'], 
                            value_vars=['Accuracy', 'Precision', 'F1 Score'],
                            var_name='Metric',
                            value_name='Score')
    
    # Create grouped bar plot
    fig = plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(data=metrics_melted, 
                          x='Model', 
                          y='Score', 
                          hue='Metric',
                          palette='viridis')
    
    # Customize the plot
    plt.title('Model Performance Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Metrics', title_fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt='%.3f', padding=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    plt.close()
    
    # Plot confusion matrices
    st.header('Confusion Matrices')
    for name, result in results.items():
        st.subheader(f'{name} Confusion Matrix')
        fig = plot_confusion_matrix(y_test, result['predictions'], 
                                  f'Confusion Matrix - {name}')
        st.pyplot(fig)
        plt.close()
    
    # Feature importance for Random Forest
    st.header('Feature Importance (Random Forest)')
    rf_model = models['Random Forest']
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    st.pyplot(fig)
    
    # Interactive Prediction
    st.header('Make New Predictions')
    
    # Create input fields for each feature
    st.subheader('Enter Wine Properties:')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.number_input('Fixed Acidity', value=7.0)
        volatile_acidity = st.number_input('Volatile Acidity', value=0.3)
        citric_acid = st.number_input('Citric Acid', value=0.3)
        residual_sugar = st.number_input('Residual Sugar', value=5.0)
        chlorides = st.number_input('Chlorides', value=0.05)
        free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=30.0)
    
    with col2:
        total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=100.0)
        density = st.number_input('Density', value=0.996)
        pH = st.number_input('pH', value=3.2)
        sulphates = st.number_input('Sulphates', value=0.5)
        alcohol = st.number_input('Alcohol', value=10.0)
    
    # Make prediction
    if st.button('Predict Wine Quality'):
        # Prepare input data
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make predictions with all models
        st.subheader('Predictions:')
        for name, model in models.items():
            prediction = model.predict(input_scaled)
            st.write(f'{name}: Quality Score = {prediction[0]}')

if __name__ == '__main__':
    main()