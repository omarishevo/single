import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file and convert it to numeric."""
    df = pd.read_csv(file_path, index_col=0)
    # Ensure all data is numeric, handling errors as NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

@st.cache_data
def plot_correlation_heatmap(df):
    """Plot a correlation heatmap of the gene expression data."""
    # Calculate the correlation matrix
    corr = df.corr()
    
    # Plot the correlation heatmap using Pandas plot method
    st.write("Correlation Heatmap of Gene Expression")
    st.dataframe(corr)  # Display the correlation matrix as a dataframe
    
    # Plot the correlation matrix as a heatmap using Streamlit's built-in functionality
    st.line_chart(corr)

@st.cache_data
def plot_line_graph(df):
    """Plot a line graph of the expression of selected genes over time or samples."""
    # Select top 10 genes for the line graph (as an example)
    selected_genes = df.iloc[:10, :].T
    
    # Plot a line graph for selected genes using Streamlit
    st.write("Gene Expression Line Graph")
    st.line_chart(selected_genes)

# Streamlit App
def main():
    st.title("Single Cell Omics Analyzer")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        st.success("✅ Data loaded successfully!")

        # Display options to plot different visualizations
        st.subheader("Choose a plot to display:")

        # Plot Correlation Heatmap
        if st.button("Plot Correlation Heatmap"):
            plot_correlation_heatmap(df)

        # Plot Line Graph for Selected Genes
        if st.button("Plot Gene Expression Line Graph"):
            plot_line_graph(df)

if __name__ == "__main__":
    main()
