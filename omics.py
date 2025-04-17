import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file and convert it to numeric."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, ignoring errors
    return df

@st.cache_data
def plot_heatmap(df):
    """Plot a heatmap of the gene expression data."""
    plt.figure(figsize=(10, 8))
    # Plotting using matplotlib's imshow to create a heatmap
    plt.imshow(df.iloc[:20, :20], cmap='viridis', aspect='auto')
    plt.colorbar(label="Expression")
    plt.title("Heatmap of Gene Expressions")
    st.pyplot(plt)

@st.cache_data
def plot_violin(df):
    """Plot a violin plot of the expression of selected genes."""
    # Select top 10 genes for the plot (as an example)
    selected_genes = df.iloc[:10, :].T

    # Plotting using matplotlib's boxplot as a substitute for violin plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(selected_genes.values, vert=False, patch_artist=True)
    plt.yticks(range(1, len(selected_genes.columns) + 1), selected_genes.columns)
    plt.title("Box Plot of Gene Expression")
    st.pyplot(plt)

@st.cache_data
def plot_bar(df):
    """Plot a bar chart of the mean expression for each gene."""
    mean_expression = df.mean(axis=1)

    plt.figure(figsize=(10, 6))
    mean_expression.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title("Mean Expression of Genes")
    plt.ylabel("Mean Expression")
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")

        # Plot Heatmap
        if st.button("Plot Heatmap"):
            plot_heatmap(df)

        # Plot Violin Plot (substituted with box plot)
        if st.button("Plot Box Plot"):
            plot_violin(df)

        # Plot Bar Chart of Mean Expression
        if st.button("Plot Bar Chart of Mean Expression"):
            plot_bar(df)

if __name__ == "__main__":
    main()
