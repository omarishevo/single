import pandas as pd
import streamlit as st
import plotly.express as px

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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data = scaler.fit_transform(df.T)
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Cell"] = df.columns

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
