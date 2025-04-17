import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tkinter import Tk, filedialog, Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SingleCellApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single Cell Omics Analyzer")
        self.df = None

        Label(root, text="Single Cell Omics GUI Tool", font=("Helvetica", 16)).pack(pady=10)

        Button(root, text="Load CSV", command=self.load_data).pack(pady=5)
        Button(root, text="Plot Heatmap", command=self.plot_heatmap).pack(pady=5)
        Button(root, text="Violin Plot", command=self.plot_violin).pack(pady=5)
        Button(root, text="PCA Plot", command=self.plot_pca).pack(pady=5)
        Button(root, text="KMeans Clustering", command=self.apply_kmeans).pack(pady=5)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path, index_col=0)
            self.df = self.df.apply(pd.to_numeric)
            print("✅ Data loaded successfully!")

    def plot_heatmap(self):
        if self.df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(self.df.iloc[:20, :20], cmap="viridis", ax=ax)
            self.display_plot(fig)

    def plot_violin(self):
        if self.df is not None:
            subset = self.df.loc[["Gene_1", "Gene_2", "Gene_3"]].T
            melted = subset.melt(var_name="Gene", value_name="Expression")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(x="Gene", y="Expression", data=melted, inner="box", ax=ax)
            self.display_plot(fig)

    def plot_pca(self):
        if self.df is not None:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.df.T)
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)
            pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
            pca_df["Cell"] = self.df.columns

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", ax=ax)
            ax.set_title("PCA of Cells")
            self.display_plot(fig)

    def apply_kmeans(self):
        if self.df is not None:
            data = StandardScaler().fit_transform(self.df.T)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(data)

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(data)
            cluster_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
            cluster_df["Cluster"] = clusters

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
            ax.set_title("KMeans Clustering on Cells")
            self.display_plot(fig)

    def display_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

# --- Run GUI ---
if __name__ == "__main__":
    root = Tk()
    app = SingleCellApp(root)
    root.mainloop()

