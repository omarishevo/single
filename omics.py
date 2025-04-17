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
    """Plot a heatmap of the gene expression data."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.iloc[:20, :20], cmap='viridis', annot=True, fmt=".2f")
    st.pyplot(plt)

@st.cache_data
def plot_violin(df):
    """Plot a violin plot of the expression of selected genes."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    selected_genes = df.iloc[:10, :].T
    selected_genes = selected_genes.melt(var_name="Gene", value_name="Expression")
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Gene", y="Expression", data=selected_genes)
    plt.title("Violin Plot of Gene Expression")
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

        # Plot Violin Plot
        if st.button("Plot Violin Plot"):
            plot_violin(df)

        # Plot Bar Chart of Mean Expression
        if st.button("Plot Bar Chart of Mean Expression"):
            plot_bar(df)

if __name__ == "__main__":
    main()
