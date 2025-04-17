import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file and convert it to numeric."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, ignoring errors
    return df

@st.cache_data
def plot_heatmap(df):
    """Plot a heatmap of the gene expression data using Streamlit."""
    # Using Streamlit's built-in ability to display the DataFrame as a heatmap
    st.write("Heatmap of Gene Expressions")
    st.dataframe(df.iloc[:20, :20])  # Show a slice of the heatmap data in tabular format

@st.cache_data
def plot_violin(df):
    """Plot a violin plot of the expression of selected genes."""
    # Select top 10 genes for the plot (as an example)
    selected_genes = df.iloc[:10, :].T

    # Using Streamlit to visualize the selected gene expressions
    st.write("Box Plot of Gene Expressions")
    st.bar_chart(selected_genes.mean(axis=1))

@st.cache_data
def plot_bar(df):
    """Plot a bar chart of the mean expression for each gene."""
    mean_expression = df.mean(axis=1)

    # Using Streamlit's built-in bar chart functionality
    st.write("Bar Chart of Mean Gene Expression")
    st.bar_chart(mean_expression)
