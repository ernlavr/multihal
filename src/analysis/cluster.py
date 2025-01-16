import sklearn
import numpy as np
import sklearn.cluster as skCluster
import sklearn.neighbors as skNeighbors
import sklearn.metrics as skMetrics
import matplotlib.pyplot as plt

def get_clustering_algorithm(args):
    if args.clustering_algo == 'kmeans':
        return Kmeans
    elif args.clustering_algo == 'dbscan':
        return Dbscan

class ClusterSuperclass():
    def __init__(self, data):
        self.data = data
        self.embeddings = np.array(data['embeddings'].to_list())

    def get_silhouette_plot(self, labels):
        # Compute the silhouette scores for each sample
        matrix = self.embeddings
        sil_avg = skMetrics.silhouette_score(matrix, labels)
        sample_sil_val = skMetrics.silhouette_samples(matrix, labels)
        num_clusters = len(list(set(labels)))
        y_lower = 10
        fig, ax = plt.subplots()

        for i in range(num_clusters):
            ith_cluster_sv = sample_sil_val[labels == i]
            ith_cluster_sv.sort()

            size_cluster_i = ith_cluster_sv.shape[0]
            y_upper = y_lower + size_cluster_i
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sv, alpha=0.7,
                label=f'Cluster {i + 1}'
            )
            y_lower = y_upper + 10  # Add space between silhouette plots

        ax.axvline(x=sil_avg, color="red", linestyle="--", label="Average Silhouette Score")
        ax.set_title(f"Silhouette Plot for {self.__class__.__name__} Clusters; n: {num_clusters}")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster")
        ax.legend(loc="upper right")
        ax.grid(True)
        return fig, ax

    def compute_silhouette(self, labels):
        sil_avg = skMetrics.silhouette_score(self.embeddings, labels, metric='cosine')
        return round(sil_avg, 4)


class Kmeans(ClusterSuperclass):
    def perform_sweep(self, data):
        num_clust = self.data['domains'].unique()

    def fit_predict(self, num_k):
        labels = skCluster.KMeans(n_clusters=num_k).fit_predict(self.embeddings)
        return labels
    
    def get_analysis_metrics(self, max_k=50):
        """ Returns a dict with all relevant analysis metrics for 1:max_k clusters"""
        output = {"dist": [], "silhouette": [-1]} # leave -1 in silh. for 1 cluster entry
        K = range(2, max_k)
        for k in K:
            kmeanModel = skCluster.KMeans(n_clusters=k)
            kmeanModel.fit(self.embeddings)
            output['dist'].append(kmeanModel.inertia_)

            labels = self.fit_predict(k)
            if k > 1:
                output['silhouette'].append(self.compute_silhouette(labels))
        
        return output

    def make_elbow_fig(self, data: dict):
        """ Data must containt key 'dist' computed from get_distortions """
        distortions = data['dist']
        K = range(1, len(distortions) + 1) # len inclusive
        silhouette = data['silhouette']
        # get value and index of silhouette
        val, k = max(silhouette), np.argmax(silhouette)
        
        fig, ax = plt.subplots()
        ax.plot(K, distortions, 'bx-')
        ax.grid(True)
        ax.set_title(f'Elbow curve Kmeans; Silhouette={val}, k={k}')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Distortion')
        return fig, ax

class Dbscan(ClusterSuperclass):
    def perform_sweep(self):
        eps_values = np.arange(0.15, 0.8, 0.05)
        min_samples = np.arange(100, 500, 25)
        output = []

        for eps in eps_values:
            for ms in min_samples:
                labels = self.fit(eps, ms)
                if len(list(set(labels))) < 2:
                    continue
                else:
                    metrics = self.get_metrics(labels)
                    output.append((metrics, eps, ms, labels))
        return np.array(output)
    
    def fit_predict(self, eps=0.68, min_samples=25):
        labels = skCluster.DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(self.embeddings)
        return labels
    
    def get_analysis_metrics(self):
        output = {"dist": None, 'silhouette': None}
        neighbors = skNeighbors.NearestNeighbors(n_neighbors=25, metric='cosine')
        neighbors_fit = neighbors.fit(self.embeddings)
        distances, indices = neighbors_fit.kneighbors(self.embeddings)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        labels = self.fit_predict()

        output['dist'] = distances
        output['silhouette'] = self.compute_silhouette(labels)
        output['clusters'] = len(np.unique(labels))
        return output
    
    def make_elbow_fig(self, data: dict):
        distances = data['dist']
        silhouette = data['silhouette']
        k = data['clusters']

        fig, ax = plt.subplots()
        ax.plot(distances)
        ax.grid(True)
        ax.set_title(f'Elbow curve DBscan; Silhouette={silhouette}, k={k}')
        ax.set_xlabel('Data point')
        ax.set_ylabel('Epsilon')
        return fig, ax

class Agglomerative(ClusterSuperclass):
    def perform_sweep(self, data):
        pass
    
    def fit(self, n_clusters=None):
        labels = skCluster.AgglomerativeClustering(n_clusters=3).fit_predict(self.embeddings)
        return labels