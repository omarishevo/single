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
def plot_heatmap(df):
    """Plot a heatmap of the gene expression data using Streamlit."""
    st.write("Heatmap of Gene Expressions")
    # Slice the first 20 rows and columns for the heatmap display
    st.dataframe(df.iloc[:20, :20])

@st.cache_data
def plot_violin(df):
    """Plot a violin plot of the expression of selected genes."""
    # Select top 10 genes for the plot (as an example)
    selected_genes = df.iloc[:10, :].T
    # Using Streamlit to visualize the selected gene expressions
    st.write("Bar Chart of Gene Expression Means")
    st.bar_chart(selected_genes.mean(axis=1))

@st.cache_data
def plot_bar(df):
    """Plot a bar chart of the mean expression for each gene."""
    mean_expression = df.mean(axis=1)
    # Using Streamlit's built-in bar chart functionality
    st.write("Bar Chart of Mean Gene Expression")
    st.bar_chart(mean_expression)

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

        # Plot Heatmap
        if st.button("Plot Heatmap"):
            plot_heatmap(df)

        # Plot Bar Chart for Selected Genes (substitute for the violin plot)
        if st.button("Plot Gene Expression Bar Chart"):
            plot_violin(df)

        # Plot Bar Chart of Mean Expression
        if st.button("Plot Bar Chart of Mean Expression"):
            plot_bar(df)

if __name__ == "__main__":
    main()
