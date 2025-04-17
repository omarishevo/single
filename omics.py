import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric)  # Ensure all data is numeric
    return df

@st.cache_data
def pca(df):
    """Perform PCA manually (using covariance matrix and eigen decomposition)."""
    # Center the data (subtract the mean from each column)
    df_centered = df - df.mean(axis=0)

    # Calculate covariance matrix
    covariance_matrix = df_centered.cov()

    # Eigenvalue and Eigenvector calculation (using numpy)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Select the top 2 eigenvectors
    pca_result = df_centered.dot(eigenvectors_sorted[:, :2])

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

    return pca_df

@st.cache_data
def apply_kmeans(df, n_clusters=3):
    """Perform KMeans clustering manually and plot results."""
    # Center the data (subtract the mean from each column)
    df_centered = df - df.mean(axis=0)

    # Randomly initialize centroids from the data (3 random rows as centroids)
    centroids = df_centered.sample(n=n_clusters, axis=1).values.T

    prev_centroids = np.zeros_like(centroids)  # Initialize previous centroids as zeros
    clusters = np.zeros(df_centered.shape[0])  # Initialize cluster assignments

    while True:
        # Calculate the distances from the data points to the centroids
        distances = np.linalg.norm(df_centered.T.values[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the nearest centroid
        new_clusters = distances.argmin(axis=1)

        # If the cluster assignments don't change, the algorithm has converged
        if np.array_equal(new_clusters, clusters):
            break

        # Update clusters and centroids
        clusters = new_clusters

        # Update centroids by averaging the points in each cluster
        new_centroids = np.array([df_centered.iloc[:, clusters == i].mean(axis=1) for i in range(n_clusters)]).T

        # Check if centroids have converged
        if np.allclose(new_centroids, prev_centroids):
            break

        prev_centroids = new_centroids

    # Plot the PCA result with clusters
    pca_df = pca(df)
    pca_df["Cluster"] = clusters

    return pca_df, centroids

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")

        # Plot PCA
        if st.button("Plot PCA"):
            pca_df = pca(df)
            st.write("### PCA Plot")
            st.write("Visualizing the first two principal components of the dataset.")
            st.scatter_chart(pca_df)  # Plot PCA in 2D

        # Apply KMeans and plot with PCA results
        if st.button("Apply KMeans Clustering"):
            pca_df, centroids = apply_kmeans(df)
            st.write("### PCA with KMeans Clusters")
            st.write("Visualizing the PCA results with KMeans clustering.")
            st.scatter_chart(pca_df[["PC1", "PC2"]])  # Plot PCA in 2D with clusters

            # Optionally display the centroids
            st.write("### KMeans Centroids")
            st.write("Centroids of the KMeans clusters in PCA space:")
            st.write(centroids)

if __name__ == "__main__":
    main()
