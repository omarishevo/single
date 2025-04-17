import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors
    return df

@st.cache_data
def plot_heatmap(df):
    st.write("Heatmap of Data (Top 20 rows and columns)")
    st.line_chart(df.iloc[:20, :20])  # Consider using a heatmap library for actual heatmaps

@st.cache_data
def plot_violin(df):
    st.write("Violin Plot of Gene Expressions (sample)")
    subset = df.loc[["Gene_1", "Gene_2", "Gene_3"]].T  # Ensure these genes exist in df
    st.line_chart(subset)

@st.cache_data
def plot_pca(df):
    st.write("PCA of Cells")
    
    # Center the data
    centered_data = df - df.mean(axis=0)

    # For simplicity, let's return the first two columns as "PCA" components
    pca_df = pd.DataFrame(centered_data.iloc[:, :2], columns=["PC1", "PC2"])
    pca_df["Cell"] = df.columns

    st.line_chart(pca_df[['PC1', 'PC2']])

def random_centroids(k, data):
    """Randomly initialize centroids from the data."""
    return [data[i] for i in range(k)] if len(data) >= k else []

def assign_clusters(centroids, data):
    """Assign each point to the nearest centroid."""
    clusters = []
    for point in data:
        distances = [sum((point[i] - centroid[i])**2 for i in range(len(point)))**0.5 for centroid in centroids]
        clusters.append(distances.index(min(distances)))
    return clusters

def recalculate_centroids(k, clusters, data):
    """Recalculate centroids based on current clusters."""
    centroids = []
    for cluster_idx in range(k):
        cluster_points = [data[i] for i in range(len(data)) if clusters[i] == cluster_idx]
        if cluster_points:
            new_centroid = [sum(point[i] for point in cluster_points) / len(cluster_points) for i in range(len(cluster_points[0]))]
            centroids.append(new_centroid)
        else:
            centroids.append([0] * len(data[0]))  # If empty, assign a zero vector
    return centroids

@st.cache_data
def apply_kmeans(df):
    st.write("KMeans Clustering on Cells")
    
    k = 3
    data = df.T.values.tolist()  # Convert DataFrame to list of lists
    centroids = random_centroids(k, data)

    if not centroids:
        st.error("Not enough data points to initialize centroids.")
        return

    prev_centroids = None
    clusters = []

    while centroids != prev_centroids:
        prev_centroids = centroids
        clusters = assign_clusters(centroids, data)
        centroids = recalculate_centroids(k, clusters, data)

    # After KMeans convergence
    if centroids:
        cluster_df = pd.DataFrame(centroids, columns=["PC1", "PC2"])
        cluster_df["Cluster"] = range(k)
        st.line_chart(cluster_df[['PC1', 'PC2']])
    else:
        st.error("No centroids were computed, please check your data.")

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.write(df.head()) 

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
