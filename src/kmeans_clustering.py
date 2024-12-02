import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA


# Функция для загрузки данных
def load_data():
    print("Загрузка данных из data/processed/processed_data.csv...")
    df = pd.read_csv('data/processed/processed_data.csv')
    return df


# Функция для выбора оптимального количества кластеров с использованием метода локтя
def plot_elbow_method(data):
    print("Определение оптимального количества кластеров...")
    distortions = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[['Age', 'Annual_Income', 'Spending_Score']])
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, distortions, marker='o')
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Distortion', fontsize=12)
    plt.grid(True)
    plt.savefig('data/processed/elbow_method.png')
    print("График метода локтя сохранён в 'data/processed/elbow_method.png'")
    plt.show()


# Функция для выполнения кластеризации с использованием KMeans
def perform_kmeans_clustering(data, n_clusters=5):
    print(f"Выполнение кластеризации K-means с {n_clusters} кластерами...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['Age', 'Annual_Income', 'Spending_Score']])

    # Сохранение модели
    joblib.dump(kmeans, 'data/processed/kmeans_model.pkl')
    print("Модель K-means сохранена в 'data/processed/kmeans_model.pkl'")
    return data, kmeans


# График парных распределений
def plot_pairwise(data):
    print("Создание парных графиков распределения...")
    sns.pairplot(data, hue='cluster', vars=['Age', 'Annual_Income', 'Spending_Score'], palette='Set2')
    plt.savefig('data/processed/pairwise_plot.png')
    print("Парный график сохранён в 'data/processed/pairwise_plot.png'")
    plt.show()


# Функция для визуализации кластеров с центроидами
def visualize_clusters(data, kmeans):
    print("Создание графика кластеров...")
    plt.figure(figsize=(10, 6))
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(
            cluster_data['Annual_Income'],
            cluster_data['Spending_Score'],
            label=f'Cluster {cluster}'
        )
    # Отображение центроидов
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 2], s=300, c='red', marker='x', label='Centroids')

    # Настройка графика
    plt.title('Customer Segmentation with Centroids', fontsize=16)
    plt.xlabel('Annual Income', fontsize=12)
    plt.ylabel('Spending Score', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Сохранение графика
    plt.savefig('data/processed/clusters_with_centroids.png')
    print("График кластеров с центроидами сохранён в 'data/processed/clusters_with_centroids.png'")
    plt.show()


# Дополнительные графики

def plot_histograms(data):
    print("Создание гистограмм для признаков...")
    features = ['Age', 'Annual_Income', 'Spending_Score']
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data, x=feature, hue='cluster', multiple="stack", palette="Set2", bins=20)
        plt.title(f'Распределение {feature} по кластерам', fontsize=16)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.grid(True)
        plt.savefig(f'data/processed/{feature}_distribution.png')
        print(f"Гистограмма для {feature} сохранена в 'data/processed/{feature}_distribution.png'")
        plt.show()


def plot_correlation_heatmap(data):
    print("Создание тепловой карты корреляций...")
    corr = data[['Age', 'Annual_Income', 'Spending_Score']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Корреляция признаков', fontsize=16)
    plt.savefig('data/processed/correlation_heatmap.png')
    print("Тепловая карта сохранена в 'data/processed/correlation_heatmap.png'")
    plt.show()


def plot_pca(data, kmeans):
    print("Понижение размерности с использованием PCA для визуализации...")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data[['Age', 'Annual_Income', 'Spending_Score']])

    pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = data['cluster']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='Set2', s=100)
    plt.title('PCA: Клиенты по кластерам', fontsize=16)
    plt.xlabel('PCA1', fontsize=12)
    plt.ylabel('PCA2', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('data/processed/pca_visualization.png')
    print("PCA график сохранён в 'data/processed/pca_visualization.png'")
    plt.show()


def plot_clusters_by_income_and_age(data):
    print("Визуализация кластеров по возрасту и доходу...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Annual_Income', hue='cluster', data=data, palette='Set2', s=100)
    plt.title('Распределение кластеров по возрасту и доходу', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Annual Income', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('data/processed/clusters_by_age_and_income.png')
    print("График кластеров по возрасту и доходу сохранён в 'data/processed/clusters_by_age_and_income.png'")
    plt.show()


def plot_cluster_sizes(data):
    print("Визуализация размера кластеров...")
    cluster_sizes = data['cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    cluster_sizes.plot(kind='bar', color='skyblue')
    plt.title('Количество клиентов в каждом кластере', fontsize=16)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Количество клиентов', fontsize=12)
    plt.grid(True)
    plt.savefig('data/processed/cluster_sizes.png')
    print("График количества клиентов в каждом кластере сохранён в 'data/processed/cluster_sizes.png'")
    plt.show()


# Основная функция
def main():
    data = load_data()

    plot_elbow_method(data)

    clustered_data, kmeans = perform_kmeans_clustering(data, n_clusters=5)

    clustered_data.to_csv('data/processed/clustered_data.csv', index=False)
    print("Кластеризация завершена! Результаты сохранены в data/processed/clustered_data.csv")

    plot_pairwise(clustered_data)

    visualize_clusters(clustered_data, kmeans)

    # Дополнительные графики
    plot_histograms(clustered_data)  # Гистограммы
    plot_correlation_heatmap(clustered_data)  # Тепловая карта корреляций
    plot_pca(clustered_data, kmeans)  # PCA для 2D визуализации
    plot_clusters_by_income_and_age(clustered_data)  # Распределение по возрасту и доходу
    plot_cluster_sizes(clustered_data)  # Количество клиентов в каждом кластере


if __name__ == "__main__":
    main()
