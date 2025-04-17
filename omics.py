import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric)
    return df

@st.cache_data
def plot_heatmap(df):
    fig = px.imshow(df.iloc[:20, :20], color_continuous_scale="Viridis", title="Heatmap of Data")
    st.plotly_chart(fig)

@st.cache_data
def plot_violin(df):
    subset = df.loc[["Gene_1", "Gene_2", "Gene_3"]].T
    melted = subset.melt(var_name="Gene", value_name="Expression")
    
    fig = px.violin(melted, x="Gene", y="Expression", box=True, title="Violin Plot of Gene Expressions")
    st.plotly_chart(fig)

@st.cache_data
def plot_pca(df):
    # Ensure df is transposed if necessary so that we perform PCA on samples (rows)
    if df.shape[0] > df.shape[1]:  # If more genes (features) than samples (cells), transpose
        df = df.T

    # Standardize the data: subtract the mean and divide by the standard deviation
    df_standardized = (df - df.mean(axis=0)) / df.std(axis=0)

    # Perform PCA manually (simplified version)
    covariance_matrix = df_standardized.cov()  # Compute covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)  # Eigenvalue decomposition

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 2 eigenvectors (for 2D PCA)
    top_eigenvectors = eigenvectors[:, :2]
    pca_result = df_standardized.dot(top_eigenvectors)  # Project the data into the PCA space

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["Cell"] = df.index  # Ensure the row indices (cells) match the PCA result

    # Plot the PCA results
    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA of Cells")
    st.plotly_chart(fig)

@st.cache_data
def apply_kmeans(df):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = StandardScaler().fit_transform(df.T)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    cluster_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    cluster_df["Cluster"] = clusters

    fig = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster", title="KMeans Clustering on Cells")
    st.plotly_chart(fig)

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
            plot_pca(df)

        if st.button("Apply KMeans Clustering"):
            apply_kmeans(df)

if __name__ == "__main__":
    main()
