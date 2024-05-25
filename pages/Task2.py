import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# Function to preprocess the data
def preprocess_data(df, target_column=None, scaler=None):
    df = df.dropna()
    features = df.columns.drop(target_column, errors='ignore')

    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])

    return df, scaler


# Function to classify a given data point
def classify_data_point(data_point, model, scaler):
    data_point = np.array(data_point).reshape(1, -1)
    data_point = scaler.transform(data_point)
    prediction = model.predict(data_point)[0]
    explanation = f"The data point is classified as {prediction} based on the model's prediction."
    return prediction, explanation


# Streamlit application
st.title('Classification Application')

st.sidebar.header('Upload your dataset')
uploaded_train_file = st.sidebar.file_uploader("Upload your training dataset (CSV)", type=["csv"])
uploaded_test_file = st.sidebar.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_train_file is not None and uploaded_test_file is not None:
    # Load datasets
    train_df = pd.read_csv(uploaded_train_file)
    test_df = pd.read_csv(uploaded_test_file)

    st.write("Training Data Preview:")
    st.write(train_df.head())
    st.write("Test Data Preview:")
    st.write(test_df.head())

    # Ensure 'target' column exists in the training dataset
    if 'target' not in train_df.columns:
        st.error("The training dataset does not contain a 'target' column.")
    else:
        # Preprocess the data
        train_df, scaler = preprocess_data(train_df, target_column='target')
        test_df, _ = preprocess_data(test_df, scaler=scaler)

        # Split the data into features and target
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df

        # Choose and train the classification algorithm (e.g., RandomForestClassifier)
        n_estimators = st.sidebar.slider('Number of estimators', 10, 200, 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on the training and test datasets
        y_train_pred = clf.predict(X_train)

        # Assuming we do not have the actual target for test data
        y_test_pred = clf.predict(X_test)

        # Evaluate the classification on the training dataset
        train_accuracy = accuracy_score(y_train, y_train_pred)
        st.write(f'Training Accuracy: {train_accuracy}')
        st.write('Classification Report for Training Data:')
        st.write(classification_report(y_train, y_train_pred))

        # Visualize the classification (using PCA for dimensionality reduction)
        pca = PCA(n_components=2)
        train_pca = pca.fit_transform(X_train)
        test_pca = pca.transform(X_test)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=y_train, palette='viridis', ax=ax[0])
        ax[0].set_title('Train Dataset Classification')
        sns.scatterplot(x=test_pca[:, 0], y=test_pca[:, 1], hue=y_test_pred, palette='viridis', ax=ax[1])
        ax[1].set_title('Test Dataset Classification')
        st.pyplot(fig)

        # Display the confusion matrix for training data
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(clf, X_train, y_train, ax=ax)
        st.pyplot(fig)

        # Functionality to classify a user-provided data point
        st.sidebar.header('Classify a Data Point')
        sample_data = []
        for feature in X_train.columns:
            value = st.sidebar.number_input(f'{feature}', value=float(X_train[feature].mean()))
            sample_data.append(value)

        if st.sidebar.button('Classify'):
            prediction, explanation = classify_data_point(sample_data, clf, scaler)
            st.sidebar.write(f'The data point is classified as: {prediction}')
            st.sidebar.write(explanation)
else:
    st.write("Please upload both training and test datasets.")
