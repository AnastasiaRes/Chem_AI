import sys
from os.path import dirname
print(sys.path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import norm
# Загрузка данных
df = pd.read_csv('psyche_drug')

# Средние значения данных
mean_values = df.mean(numeric_only=True)
std_values = df.std(numeric_only=True)

print(mean_values)
print(std_values)

# Доверительный интервал
for column in df.columns:
    if df[column].dtype.kind in 'bifc':  # Проверяем, является ли столбец числовым
        mean = df[column].mean()
        std = df[column].std()
        n = len(df[column])
        interval = (mean - 1.96 * std / np.sqrt(n), mean + 1.96 * std / np.sqrt(n))
        print(f"Доверительный интервал для колонки '{column}': {interval}")
# Рассчитываем медиану для каждой колонки
median_values = df.median(numeric_only=True)

# Выводим медиану
print("Медиана для каждой колонки:")
print(median_values)

# максимальное и минимальное значения
max_values = df.max()
min_values = df.min()
print("Максимальные значения:")
print(max_values)

print("\nМинимальные значения:")
print(min_values)

#расчет дисперсии
print("Типы данных в каждом столбце:")
print(df.dtypes)

# Заменяем строковые значения на NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Удаляем строки с NaN, если они есть
df = df.dropna()

# Кластеризация
numeric_data = df.select_dtypes(include=[np.number])

# Нормализация данных

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def ward_linkage(cluster1, cluster2):
    """
    Вычисляет расстояние Ward's linkage между двумя кластерами.
    """
    n1 = len(cluster1)
    n2 = len(cluster2)
    mean1 = np.mean(cluster1, axis=0)
    mean2 = np.mean(cluster2, axis=0)
    mean_total = np.mean(np.concatenate((cluster1, cluster2)), axis=0)
    
    return (n1 * n2 / (n1 + n2)) * np.sum((mean1 - mean_total)**2 + (mean2 - mean_total)**2)


def hierarchical_clustering(data, n_clusters, linkage_method='ward'):
    # Нормализуем данные перед кластеризацией
    data = normalize_data(data)
    
    clusters = [[point] for point in data]

    while len(clusters) > n_clusters:
        min_distance = float('inf')
        merge_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage_method == 'ward':
                    distance = ward_linkage(clusters[i], clusters[j])
                elif linkage_method == 'single':
                    distance = single_linkage(clusters[i], clusters[j])
                else:
                    raise ValueError("Неверный метод связи")
                
                if distance < min_distance:
                    min_distance = distance
                    merge_pair = (i, j)

        # Объединяем выбранные кластеры
        cluster1, cluster2 = merge_pair
        merged_cluster = clusters[cluster1] + clusters[cluster2]
        clusters = [c for k, c in enumerate(clusters) if k not in merge_pair]
        clusters.append(merged_cluster)

    return clusters


if __name__ == "__main__": 
    
    np.random.seed(0)
    data = np.random.rand(100, 2)
    n_clusters = 5

    # Выполняем кластеризацию с использованием Ward's linkage
    result = hierarchical_clustering(data, n_clusters=5, linkage_method='ward')

    # Проверка содержимого кластеров
    print("Кластеры после кластеризации:")
    print(result)

    # Визуализация кластеров
    def visualize_clusters(clusters):
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

        plt.figure(figsize=(10, 7))

        for i, cluster in enumerate(clusters):
            cluster_points = np.array(cluster)
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')

        plt.title('Visualization of Clusters')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()

    visualize_clusters(result)

def visualize_clusters(data, hierarchical_clusters, kmeans_labels):
# Выполняем иерархическую кластеризацию
    hierarchical_result = hierarchical_clustering(data, n_clusters=n_clusters, linkage_method='ward')

    # Выполняем k-means кластеризацию
    kmeans_labels = kmeans_clustering(data, n_clusters=n_clusters)

    # Визуализация результатов
    visualize_clusters(data, hierarchical_result, kmeans_labels)

    # Выводим информацию о кластерах
    print("Иерархическая кластеризация:")
    for i, cluster in enumerate(hierarchical_result):
        print(f"Кластер {i + 1}: {len(cluster)} точек")

    print("\nK-means кластеризация:")
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Кластер {label + 1}: {count} точек")
        
        plt.title('Visualization of Clusters')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()
        import sys
from os.path import dirname
print(sys.path)


def hierarchical_clustering(data, n_clusters, linkage_method='ward'):
    # Нормализуем данные перед кластеризацией
    data = normalize_data(data)
    
    # Выполняем иерархическую кластеризацию
    linked = linkage(data, method=linkage_method)
    
    # Возвращаем связи для построения дендрограммы
    return linked

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

def plot_dendrogram(linked):
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title('Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    # Выполняем иерархическую кластеризацию
    linked = hierarchical_clustering(data, n_clusters=n_clusters, linkage_method='ward')

import sys
from os.path import dirname
print(sys.path)

def hierarchical_clustering(data, n_clusters, linkage_method='ward'):
    # Нормализуем данные перед кластеризацией
    data = normalize_data(data)
    
    # Выполняем иерархическую кластеризацию
    linked = linkage(data, method=linkage_method)
    
    # Получаем метки кластеров
    labels = fcluster(linked, n_clusters, criterion='maxclust')
    
    return linked, labels

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def plot_dendrogram(linked):
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title('Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

def plot_boxplots_with_points (data, hierarchical_labels, kmeans_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Box plot для иерархической кластеризации
    df_hierarchical = pd.DataFrame(data)
    df_hierarchical['Cluster'] = hierarchical_labels
    df_hierarchical.boxplot(column=[0, 1], by='Cluster', ax=ax1)
    ax1.set_title('Hierarchical Clustering')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Values')

    # Box plot для k-means кластеризации
    df_kmeans = pd.DataFrame(data)
    df_kmeans['Cluster'] = kmeans_labels
    df_kmeans.boxplot(column=[0, 1], by='Cluster', ax=ax2)
    ax2.set_title('K-means Clustering')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Values')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100, 2)
    n_clusters = 5

    # Выполняем иерархическую кластеризацию
    linked, hierarchical_labels = hierarchical_clustering(data, n_clusters=n_clusters, linkage_method='ward')

    # Выполняем k-means кластеризацию
    kmeans_labels = kmeans_clustering(data, n_clusters=n_clusters)

    # Визуализация результатов
    plot_dendrogram(linked)
    plot_boxplots_with_points (data, hierarchical_labels, kmeans_labels)

    # Выводим информацию о кластерах
    print("Иерархическая кластеризация:")
    unique, counts = np.unique(hierarchical_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Кластер {label}: {count} точек")

# параметры для каждого распределения
params = [
    {"mu": 0, "sigma": 1, "label": "Распределение параметров"},
]

# диапазон значений x
x = np.linspace(-5, 5, 1000)

# график
plt.figure(figsize=(10, 6))  # размер графика

for p in params:
    # значения y для каждого распределения
    y = norm.pdf(x, loc=p["mu"], scale=p["sigma"])
    
    # Построение графика
    plt.plot(x, y, label=p["label"])

# Настройка графика
plt.title("График нормального распределения")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid(True) 
# plt.show()