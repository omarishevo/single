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
def plot_heatmap(df):
    """Plot a simple heatmap using Streamlit's line_chart."""
    subset = df.iloc[:20, :20]  # Taking top 20 rows and columns for heatmap
    st.line_chart(subset)  # Simple line chart for heatmap

@st.cache_data
def plot_violin(df):
    """Plot gene expression using Streamlit's bar_chart."""
    subset = df.loc[["Gene_1", "Gene_2", "Gene_3"]].T
    melted = subset.melt(var_name="Gene", value_name="Expression")
    st.bar_chart(melted['Expression'])  # Plot expression values using a bar chart

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

    # Plot the PCA results with clusters using Streamlit's scatter_chart
    st.write("### PCA with KMeans Clusters")
    st.scatter_chart(pca_df[["PC1", "PC2"]])  # Plot PCA in 2D
    return pca_df

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")
        
        if st.button("Plot Heatmap"):
            plot_heatmap(df)

        if st.button("Plot Violin Plot"):
            plot_violin(df)

        if st.button("Plot PCA"):
            pca_df = pca(df)
            st.write("### PCA Plot")
            st.scatter_chart(pca_df[["PC1", "PC2"]])  # Plot PCA in 2D

        if st.button("Apply KMeans Clustering"):
            pca_df = apply_kmeans(df)
            st.write("### KMeans Clustering Results on PCA")
            st.scatter_chart(pca_df[["PC1", "PC2", "Cluster"]])  # Plot PCA with clusters

if __name__ == "__main__":
    main()
