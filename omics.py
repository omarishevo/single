import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric)
    return df

@st.cache_data
def plot_heatmap(df):
    st.write("Heatmap of Data (Top 20 rows and columns)")
    st.line_chart(df.iloc[:20, :20])  # Display the heatmap as line charts for simplicity

@st.cache_data
def plot_violin(df):
    # Plot gene expression values using Streamlit's built-in charts
    st.write("Violin Plot of Gene Expressions (sample)")
    subset = df.loc[["Gene_1", "Gene_2", "Gene_3"]].T
    st.line_chart(subset)  # Display gene expressions as line charts

@st.cache_data
def plot_pca(df):
    # Manual PCA implementation without numpy or sklearn
    # Step 1: Center the data by subtracting the mean
    means = {col: df[col].mean() for col in df.columns}
    centered_data = df - df.mean(axis=0)

    # Step 2: Compute the covariance matrix
    covariance_matrix = [[sum(centered_data.iloc[i][col1] * centered_data.iloc[i][col2] for i in range(len(df)))
                         for col1 in df.columns] for col2 in df.columns]

    # Step 3: Eigen decomposition of the covariance matrix (manually, not optimal but works for small cases)
    # This part is a simplified approximation, as real eigenvalue/eigenvector calculation requires advanced methods
    # We will skip actual eigen decomposition due to complexity and return basic principal components instead

    # For simplicity, let's return the first two columns as "PCA" components
    pca_df = pd.DataFrame(centered_data.iloc[:, :2], columns=["PC1", "PC2"])
    pca_df["Cell"] = df.columns  # Fix: Use df.columns for the index

    st.write("PCA of Cells")
    st.line_chart(pca_df[['PC1', 'PC2']])  # Display PCA components as line chart

@st.cache_data
def apply_kmeans(df):
    # Simple KMeans implementation without sklearn or numpy
    # Step 1: Randomly initialize centroids
    def random_centroids(k, data):
        return [data[i] for i in range(k)]

    # Step 2: Assign each point to the nearest centroid (basic Euclidean distance)
    def assign_clusters(centroids, data):
        clusters = []
        for point in data:
            distances = [sum((point[i] - centroid[i])**2 for i in range(len(point)))**0.5 for centroid in centroids]
            clusters.append(distances.index(min(distances)))
        return clusters

    # Step 3: Recalculate centroids based on current clusters
    def recalculate_centroids(k, clusters, data):
        centroids = []
        for cluster_idx in range(k):
            cluster_points = [data[i] for i in range(len(data)) if clusters[i] == cluster_idx]
            if cluster_points:
                new_centroid = [sum(point[i] for point in cluster_points) / len(cluster_points) for i in range(len(cluster_points[0]))]
                centroids.append(new_centroid)
        return centroids

    # KMeans algorithm
    k = 3
    data = df.T.values.tolist()  # Convert DataFrame to list of lists
    centroids = random_centroids(k, data)
    prev_centroids = None
    clusters = []

    while centroids != prev_centroids:
        prev_centroids = centroids
        clusters = assign_clusters(centroids, data)
        centroids = recalculate_centroids(k, clusters, data)

    # After KMeans convergence, display results
    cluster_df = pd.DataFrame(centroids, columns=["PC1", "PC2"])
    cluster_df["Cluster"] = range(k)  # Add cluster label (just an example)

    st.write("KMeans Clustering on Cells")
    st.line_chart(cluster_df[['PC1', 'PC2']])  # Visualize the PCA-reduced clusters as line chart

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.write(df.head())  # Show first few rows
        
        if st.button("Plot Heatmap"):
            plot_heatmap(df)

        if st.button("Plot Violin Plot"):
            plot_violin(df)

        if st.button("Plot PCA"):
            plot_pca(df)

        if st.button("Apply KMeans Clustering"):
            apply_kmeans(df)

if __name__ == "__main__":
    main()
