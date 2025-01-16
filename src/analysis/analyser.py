import polars as pl
import matplotlib.pyplot as plt
import os
import umap
import logging
import numpy as np
import sklearn.cluster as skCluster
import sklearn.manifold as skManifold
import sklearn.decomposition as skDecomp
import sklearn.cluster as skCluster
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import src.utils.wandb_manager as wbMang
import src.analysis.cluster as MultihalCluster
import src.analysis.figures as figs
import src.utils.config as config

from tqdm import tqdm

class DatasetAnalyser():
    def __init__(self, df: pl.DataFrame, args):
        self.df = df
        self.args = args

    def _get_embedding_matrix(self, data):
        return np.array(data['embeddings'].to_list())
    
    def remove_duplicates_by_cossim(self, data, threshold=0.99):
        # Compute the cosine similarity between the embeddings
        matrix = self._get_embedding_matrix(data)
        cos_sim = cosine_similarity(matrix)

        rows, cols = np.where(cos_sim > threshold)
        mask = rows != cols # ignore the diagonal
        rows = rows[mask]
        cols = cols[mask]

        # Filter flipped rows and cols
        row_, col_ = [], []
        seen = set()
        for r, c in zip(rows, cols):
            pair = tuple(sorted((r, c)))
            if pair not in seen and cos_sim[r, c] > threshold:
                seen.add(pair)
                row_.append(r.item())
                col_.append(c.item())

        df_tmp = self.df
        cols_to_map = ['output', 'optional_output', 'incorrect_answers', 'context', 'context_type']

        # merge target into source
        for source_index, target_index in zip(row_, col_):
            for col_name in cols_to_map:
                source_value = df_tmp[col_name][source_index]
                target_value = df_tmp[col_name][target_index]
                col_dtype = df_tmp[col_name].dtype

                # Combine source and target values based on type
                if col_dtype == pl.String:
                    if source_value is None and target_value is None:
                        continue # nothing to merge
                    elif source_value is None or target_value is None:
                        updated_value = source_value or target_value # Use non-null value or keep as None
                    elif source_value == target_value:
                        # If equal, use either value (no concatenation needed)
                        updated_value = source_value
                    else:
                        updated_value = source_value + config.LIST_SEP + target_value
                else:
                    raise ValueError(f"Mergin columns during duplicate removal by unsupported column type: {col_dtype}")

                # remove duplicates
                updated_value = updated_value.split(config.LIST_SEP)
                updated_value = config.LIST_SEP.join(list(set(updated_value)))

                # Update the target index value
                col_series = df_tmp[col_name]
                col_series[source_index] = updated_value

                # Replace the column in the dataframe with the updated series
                df_tmp = df_tmp.with_columns(pl.Series(col_name, col_series))


        
        # Remove the duplicate rows
        if self.args.debug_mode:
            tmp = df_tmp.with_row_index().filter(pl.col("index").is_in(col_)).drop('index')
            tmp.write_json("output/rows_removed_after_cossim.json")
            logging.info(f"Removed {len(col_)} duplicate rows based on cosine similarity; threshold: {threshold}")
            merged = list(zip(row_, col_))
            logging.info(f"Merged (src <- tgt): {merged}")

            # find longest domain str
            max_domain = max([len(df_tmp['domain'][i[0]]) for i in merged if df_tmp['domain'][i[0]]])
            max_src_ds = max([len(df_tmp['source_dataset'][i[0]]) for i in merged if df_tmp['source_dataset'][i[0]]])

            for i in merged:
                # find longest domain str
                logging.info(f"Ds: {df_tmp['source_dataset'][i[0]]:{max_src_ds}}; Domain: {df_tmp['domain'][i[0]]:{max_domain}}; Source ({df_tmp['id'][i[0]]}): {df_tmp['input'][i[0]]}")
                logging.info(f"Ds: {df_tmp['source_dataset'][i[1]]:{max_src_ds}}; Domain: {df_tmp['domain'][i[1]]:{max_domain}}; Target ({df_tmp['id'][i[0]]}): {df_tmp['input'][i[1]]}")
                logging.info("")
              


        df_tmp = df_tmp.with_row_index().filter(~pl.col("index").is_in(col_)).drop('index')


            



        pass

    def generate_clustering_evals(self, data):
        # elbow
        output_folder = f'output/figures/analysis/{self.args.clustering_algo}'

    def run_precluster_analysis(self, data):
        output_folder = f'output/figures/analysis/{self.args.clustering_algo}'
        os.makedirs(output_folder, exist_ok=True)
        clusters = data['domain'].unique().count()

        mlClustClass = MultihalCluster.get_clustering_algorithm(self.args)
        mlClust = mlClustClass(data)
        clust_metrics = mlClust.get_analysis_metrics()
        fig, ax = mlClust.make_elbow_fig(clust_metrics)
        fig.savefig(os.path.join(output_folder, 'elbow_plot.png'))

    def run_dim_red(self, data):
        pca = skDecomp.PCA(n_components=2)
        self.visualize_clusters(data, pca)

        tsne = skManifold.TSNE(n_components=2, n_iter=3000)
        self.visualize_clusters(data, tsne)

        umapModel = umap.UMAP(metric='cosine')
        self.visualize_clusters(data, umapModel)

    def cluster_by_embeddings(self, data) -> pl.DataFrame:
        mlClustClass = MultihalCluster.get_clustering_algorithm(self.args)
        mlClust = mlClustClass(data)
        labels = mlClust.fit_predict()

        # add labels to data
        data = data.with_columns(
            cluster=pl.Series('cluster', labels)
        )
        return data
        
        # cosine similarities
        self.cosine_similarity(data)
        # self.plot_silhouette(data)

        # wbMang.WandbManager().log_dataframe(data)

    def get_sentence_similarities(self, data, threshold):
        # Compute the cosine similarity between the embeddings
        matrix = self._get_embedding_matrix(data)
        cos_sim = cosine_similarity(matrix)
        rows, cols = np.where(cos_sim > -1)

        # ignore the diagonal
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]

        # Iterate through both arrays
        row_, col_ = [], []
        seen = set()
        for r, c in zip(rows, cols):
            # Create a sorted tuple to ensure uniqueness
            pair = tuple(sorted((r, c)))
            if pair not in seen and cos_sim[r, c] > threshold:
                seen.add(pair)
                row_.append(r)
                col_.append(c)

        # write to file, X(domain) - Y(domain)
        similarities = pl.DataFrame({
            'X_s': data['input'][row_],
            'Y_s': data['input'][col_],

            'X': data['domain'][row_],
            'Y': data['domain'][col_],
            
            'X_ds': data['source_dataset'][row_],
            'Y_ds': data['source_dataset'][col_],
            
            'X_ans': data['output'][col_],
            'Y_ans': data['output'][row_],

            'similarity': pl.Series('similarity', cos_sim[row_, col_])
        })

        # flip rows and cols
        cols = similarities.columns
        similarities_t = similarities.transpose()
        # add column at 0 index
        similarities_t.insert_column(0, pl.Series('cols', cols))
        
        similarities_t.write_csv("output/cos_sim_sentence_pairs.csv")
        return None

    

    def get_cosine_similarity(self, data):
        matrix = self._get_embedding_matrix(data)
        rows, cols = cosine_similarity(matrix)
        return rows, cols

    def visualize_clusters(self, data, dim_red):
        # Perform dimensionality reduction
        matrix = self._get_embedding_matrix(data)

        # Visualize the clusters
        reduced = dim_red.fit_transform(matrix)
        
        fig, ax = plt.subplots()
        ax.scatter(reduced[:, 0], 
                    reduced[:, 1], 
                    s=10, 
                    c=data['domain_encoded'].to_list(), 
                    cmap='viridis', 
                    alpha=0.7)
        
        dim_red_name = dim_red.__class__.__name__
        ax.set_title(f"Dim.Reduction: {dim_red_name}")
        fig.legend()
        output_path = f'output/figures/analysis/{dim_red_name}'
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, 'dim_red.png'))
        fig.clf()    # pass

    def plot_elbow(self, data):
        distorsions = []
        eps = np.arange(0.15, 0.6, 0.05)
        matrix = self._get_embedding_matrix(data)
        for k in tqdm(eps, 'Plotting Elbow'):
            kmeans = skCluster.DBSCAN(eps=k)
            labels = kmeans.fit_predict(matrix)
            distorsions.append(labels)
            
            # if there are less than 1 unique labels, continue
            if len(list(set(labels))) < 2:
                continue

            labels = [i + 1 for i in labels]
            data = data.with_columns(
                cluster=pl.Series('cluster', labels)
            )
            self.plot_silhouette(data)

        fig, ax = plt.subplots()
        ax.plot(eps, distorsions)
        ax.grid(True)
        ax.set_title('Elbow curve')
        os.makedirs('output/figures/analysis/', exist_ok=True)
        fig.savefig('output/figures/analysis/elbow_curve.png')
        wbMang.WandbManager().log_figure('Elbow Plot', fig)

    def plot_silhouette(self, data):
        # Compute the silhouette scores for each sample
        matrix = self._get_embedding_matrix(data)
        sil_avg = silhouette_score(matrix, data['cluster'])
        sample_sil_val = silhouette_samples(matrix, data['cluster'])
        num_clusters = len(data['cluster'].unique())
        y_lower = 10
        fig, ax = plt.subplots()

        for i in range(num_clusters):
            ith_cluster_sv = sample_sil_val[data['cluster'] == i]
            ith_cluster_sv.sort()

            size_cluster_i = ith_cluster_sv.shape[0]
            y_upper = y_lower + size_cluster_i
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sv,
                alpha=0.7,
                label=f'Cluster {i + 1}'
            )
            y_lower = y_upper + 10  # Add space between silhouette plots

        ax.axvline(x=sil_avg, color="red", linestyle="--", label="Average Silhouette Score")
        ax.set_title(f"Silhouette Plot for t-SNE Clusters; n: {num_clusters}")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster")
        ax.legend(loc="upper right")
        ax.grid(True)

        os.makedirs('output/figures/analysis/', exist_ok=True)
        fig.savefig(f'output/figures/analysis/silhouette_plot_n{num_clusters}.png')
        wbMang.WandbManager().log_figure(f'Silhouette Plot; n: {num_clusters}', fig)