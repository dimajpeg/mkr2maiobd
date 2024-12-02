import pandas as pd
from sklearn.cluster import KMeans
from multiprocessing import Pool
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, n_chunks):
    return np.array_split(data, n_chunks)

def map_cluster(data_chunk):
    kmeans = KMeans(n_clusters=5, random_state=42)
    data_chunk['cluster'] = kmeans.fit_predict(data_chunk[['Age', 'Annual_Income', 'Spending_Score']])
    return data_chunk

def reduce_centroids(mapped_chunks):
    full_data = pd.concat(mapped_chunks, ignore_index=True)
    centroids = full_data.groupby('cluster').mean()[['Age', 'Annual_Income', 'Spending_Score']]
    return centroids

if __name__ == "__main__":
    input_path = 'data/processed/processed_data.csv'

    data = load_data(input_path)

    chunks = split_data(data, n_chunks=4)

    with Pool(processes=4) as pool:
        mapped_chunks = pool.map(map_cluster, chunks)

    centroids = reduce_centroids(mapped_chunks)
    print("Центроїди кластерів:")
    print(centroids)
