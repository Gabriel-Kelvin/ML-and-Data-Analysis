import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Function to preprocess the data
def preprocess_data(df, scaler=None):
    df = df.dropna()
    features = df.columns.drop('target', errors='ignore')

    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])

    return df, scaler


# Function to identify cluster for a given data point
def identify_cluster(data_point, model, scaler):
    data_point = np.array(data_point).reshape(1, -1)
    data_point = scaler.transform(data_point)
    cluster = model.predict(data_point)[0]
    explanation = f"The data point belongs to cluster {cluster} based on the model's prediction."
    return cluster, explanation


# Streamlit application
st.title('Clustering Application')

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

    # Preprocess the data
    train_df, scaler = preprocess_data(train_df)
    test_df, _ = preprocess_data(test_df, scaler)

    # Choose and train the clustering algorithm (e.g., K-Means)
    n_clusters = st.sidebar.slider('Number of clusters', 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_df.drop('target', axis=1, errors='ignore'))

    # Predict the clusters for the training and test datasets
    train_df['Cluster'] = kmeans.predict(train_df.drop('target', axis=1, errors='ignore'))
    test_df['Cluster'] = kmeans.predict(test_df)

    # Evaluate the clustering
    sil_score = silhouette_score(train_df.drop(['target', 'Cluster'], axis=1, errors='ignore'), train_df['Cluster'])
    st.write(f'Silhouette Score: {sil_score}')

    # Visualize the clusters
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_df.drop(['target', 'Cluster'], axis=1, errors='ignore'))
    test_pca = pca.transform(test_df.drop('Cluster', axis=1, errors='ignore'))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=train_df['Cluster'], palette='viridis', ax=ax[0])
    ax[0].set_title('Train Dataset Clusters')
    sns.scatterplot(x=test_pca[:, 0], y=test_pca[:, 1], hue=test_df['Cluster'], palette='viridis', ax=ax[1])
    ax[1].set_title('Test Dataset Clusters')
    st.pyplot(fig)

    # Functionality to identify cluster for a user-provided data point
    st.sidebar.header('Identify Cluster for a Data Point')
    sample_data = []
    for feature in train_df.columns.drop(['target', 'Cluster']):
        value = st.sidebar.number_input(f'{feature}', value=float(train_df[feature].mean()))
        sample_data.append(value)

    if st.sidebar.button('Identify Cluster'):
        cluster, explanation = identify_cluster(sample_data, kmeans, scaler)
        st.sidebar.write(f'The data point belongs to cluster: {cluster}')
        st.sidebar.write(explanation)
else:
    st.write("Please upload both training and test datasets.")
