import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric)  # Convert all data to numeric, if not already
    return df

@st.cache_data
def plot_heatmap(df):
    """Plot a simple heatmap of the first 20 rows and columns using Streamlit's line_chart."""
    subset = df.iloc[:20, :20]
    st.line_chart(subset)  # Simple line chart as a heatmap alternative

@st.cache_data
def plot_violin(df):
    """Plot a simple bar chart for selected genes."""
    subset = df.loc[["Gene_1", "Gene_2", "Gene_3"]].T
    melted = subset.melt(var_name="Gene", value_name="Expression")
    
    st.bar_chart(melted['Expression'])  # Plot expression values using a bar chart

@st.cache_data
def pca(df):
    """Simplified PCA using basic covariance matrix and eigenvalue/eigenvector method."""
    # Center the data (subtract the mean from each column)
    df_centered = df - df.mean(axis=0)

    # Calculate covariance matrix
    covariance_matrix = df_centered.cov()

    # Eigenvalue and Eigenvector computation using pandas' covariance matrix
    eigenvalues, eigenvectors = pd.np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Select top 2 eigenvectors for PCA
    pca_result = df_centered.dot(eigenvectors_sorted[:, :2])

    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["Cell"] = df.columns  # Ensure alignment with original cells

    return pca_df

@st.cache_data
def apply_kmeans(df, n_clusters=3):
    """Apply KMeans clustering with basic implementation."""
    # Center the data (subtract the mean from each column)
    df_centered = df - df.mean(axis=0)

    # Initialize cluster centroids (randomly from the data)
    centroids = df.sample(n=n_clusters, axis=1).values.T

    prev_centroids = centroids.copy()
    while True:
        # Calculate Euclidean distances to centroids
        distances = ((df_centered.T - centroids[:, np.newaxis])**2).sum(axis=2)
        clusters = distances.argmin(axis=1)

        # Update centroids
        centroids = np.array([df_centered.iloc[:, clusters == i].mean(axis=1) for i in range(n_clusters)]).T

        if np.allclose(centroids, prev_centroids):
            break
        prev_centroids = centroids

    pca_df = pca(df)
    pca_df["Cluster"] = clusters

    # Simple bar chart for cluster visualization
    st.bar_chart(pca_df["Cluster"])

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
            st.line_chart(pca_df[["PC1", "PC2"]])

        if st.button("Apply KMeans Clustering"):
            apply_kmeans(df)

if __name__ == "__main__":
    main()
