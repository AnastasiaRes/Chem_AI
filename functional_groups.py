import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Путь к файлу относительно местоположения текущего скрипта
base_dir = os.path.dirname(__file__)  # Получение пути к директории, где находится текущий скрипт
file_path = os.path.join(base_dir, 'full_dataset.csv')  # Построение полного пути к файлу с расширением .csv

# Загрузка данных
df = pd.read_csv(file_path)

# List of columns with functional groups
functional_group_columns = [
    'NumLipinskiHBA', 'NumLipinskiHBD', 'NumAmideBonds', 'NumHBD', 'NumHBA',
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
    'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles'
]

# Separate data into psychedelic and non-psychedelic substances
psychedelic = df[df['is_psychedelic'] == 1]
non_psychedelic = df[df['is_psychedelic'] == 0]

# Plot histograms for each functional group column comparing the two groups
for column in functional_group_columns:
    plt.figure(figsize=(12, 6))

    # Plot normalized histogram for psychedelic substances
    plt.hist(
        psychedelic[column].dropna(), bins=30, density=True, alpha=0.5,
        label='Psychedelic', edgecolor='black'
    )

    # Plot normalized histogram for non-psychedelic substances
    plt.hist(
        non_psychedelic[column].dropna(), bins=30, density=True, alpha=0.5,
        label='Non-Psychedelic', edgecolor='black'
    )

    # Graph settings
    plt.title(f'Distribution of {column} in Psychedelic vs Non-Psychedelic')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()

    # Show the graph
 #   plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Extracting only the functional group columns
X = df[functional_group_columns].dropna()

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Adding the PCA results back to the DataFrame for plotting
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['is_psychedelic'] = df['is_psychedelic']

# Plotting the results of PCA
plt.figure(figsize=(10, 6))
plt.scatter(
    df_pca[df_pca['is_psychedelic'] == 1]['PCA1'],
    df_pca[df_pca['is_psychedelic'] == 1]['PCA2'],
    alpha=0.6, label='Psychedelic', edgecolor='black'
)
plt.scatter(
    df_pca[df_pca['is_psychedelic'] == 0]['PCA1'],
    df_pca[df_pca['is_psychedelic'] == 0]['PCA2'],
    alpha=0.6, label='Non-Psychedelic', edgecolor='black'
)

plt.title('PCA of Functional Groups: Psychedelic vs Non-Psychedelic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# t-SNE (t-distributed Stochastic Neighbor Embedding)
'''t-SNE (t-distributed Stochastic Neighbor Embedding) – это метод снижения размерности и визуализации данных, который позволяет сохранить локальные структуры данных и обнаруживать нелинейные зависимости. Основная идея заключается в том, чтобы преобразовать исходные данные таким образом, чтобы схожие объекты в исходном пространстве сохраняли свою схожесть и в новом, сниженном пространстве. Одной из ключевых особенностей t-SNE является то, что он нацелен на сохранение локальных структур данных, что делает его особенно полезным для визуализации скрытых паттернов в данных.'''
from sklearn.manifold import TSNE

# Use the same functional group columns as for PCA
X = df[functional_group_columns].dropna()

# Standardizing the data
X_scaled = scaler.fit_transform(X)

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Adding the t-SNE results back to the DataFrame for plotting
df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
df_tsne['is_psychedelic'] = df['is_psychedelic']

# Plotting the results of t-SNE
plt.figure(figsize=(10, 6))
plt.scatter(
    df_tsne[df_tsne['is_psychedelic'] == 1]['t-SNE1'],
    df_tsne[df_tsne['is_psychedelic'] == 1]['t-SNE2'],
    alpha=0.6, label='Psychedelic', edgecolor='black'
)
plt.scatter(
    df_tsne[df_tsne['is_psychedelic'] == 0]['t-SNE1'],
    df_tsne[df_tsne['is_psychedelic'] == 0]['t-SNE2'],
    alpha=0.6, label='Non-Psychedelic', edgecolor='black'
)

plt.title('t-SNE of Functional Groups: Psychedelic vs Non-Psychedelic')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()


# Используем результаты t-SNE для кластеризации
X_tsne = df_tsne[['t-SNE1', 't-SNE2']]

# Определение оптимального количества кластеров с помощью метода локтя и Silhouette Score
silhouette_scores = []
inertia = []

# Проверяем количество кластеров от 2 до 10
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_tsne)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_tsne, labels))
    inertia.append(kmeans.inertia_)

# Визуализация метода локтя
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# На основе графиков выберите оптимальное количество кластеров
# Оптимальное количество кластеров равно 4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_tsne['Cluster'] = kmeans.fit_predict(X_tsne)

# Визуализация кластеров
plt.figure(figsize=(10, 6))
plt.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'], c=df_tsne['Cluster'], cmap='viridis', alpha=0.6, edgecolor='black')
plt.title('t-SNE Clustering with K-means')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# Анализ распределения психоделических и не психоделических веществ по кластерам
cluster_counts = df_tsne.groupby('Cluster')['is_psychedelic'].value_counts().unstack().fillna(0)
cluster_counts.columns = ['Non-Psychedelic', 'Psychedelic']
print(cluster_counts)

# Построение barplot для визуализации распределения психоделических и не психоделических веществ по кластерам
cluster_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of Psychedelic vs Non-Psychedelic Substances in Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Substance Type')
plt.show()

# Определение средних значений признаков для каждого кластера
# Добавим к исходным данным информацию о кластерах
df_with_clusters = df.join(df_tsne['Cluster'])

# Рассчет средних значений функциональных групп для каждого кластера
cluster_means = df_with_clusters.groupby('Cluster')[functional_group_columns].mean()

# Добавление количества психоделических и не психоделических веществ в каждом кластере
cluster_counts = df_with_clusters.groupby('Cluster')['is_psychedelic'].value_counts().unstack().fillna(0)
cluster_counts.columns = ['Non-Psychedelic', 'Psychedelic']

# Объединение данных о средних значениях и распределении веществ в одну таблицу
cluster_summary = pd.concat([cluster_means, cluster_counts], axis=1)

# Показ таблицы в виде вывода на экран
print(cluster_summary)

#Сохранить таблицу в файл CSV для дальнейшего анализа
# cluster_summary.to_csv('cluster_analysis_summary.csv')

# АНАЛИЗ ТАБЛИЦЫ ПРО КЛАСТЕРЫ
from scipy.stats import f_oneway
import pandas as pd

# Добавление информации о кластерах к исходным данным
df_with_clusters = df.join(df_tsne['Cluster'])

# Список для хранения результатов ANOVA
anova_results = []

# Проведение ANOVA для каждого функционального признака
# Про данный метод можно прочитать вот здесь: https://habr.com/ru/companies/otus/articles/734258/
for column in functional_group_columns:
    # Группируем данные по кластерам для текущего признака
    groups = [df_with_clusters[df_with_clusters['Cluster'] == cluster][column] for cluster in
              df_with_clusters['Cluster'].unique()]

    # Выполняем ANOVA
    f_val, p_val = f_oneway(*groups)

    # Сохраняем результаты
    anova_results.append({'Feature': column, 'F-Value': f_val, 'P-Value': p_val})

# Преобразуем результаты в DataFrame и сортируем по значению P-Value
anova_df = pd.DataFrame(anova_results).sort_values(by='P-Value')

# Показать таблицу с результатами
print(anova_df)

# Сохранение результатов в файл CSV для дальнейшего анализа
#anova_df.to_csv('anova_analysis_of_functional_groups.csv', index=False)

# Интерпретация результатов
significant_features = anova_df[anova_df['P-Value'] < 0.05]  # Установим порог значимости p-value < 0.05

# Печать интерпретации для значимых признаков
print("\nИнтерпретация значимых функциональных групп (P-Value < 0.05):")
for index, row in significant_features.iterrows():
    print(f"Функциональная группа '{row['Feature']}' имеет значительные различия между кластерами "
          f"(F-Value = {row['F-Value']:.2f}, P-Value = {row['P-Value']:.4f}). "
          f"Это может указывать на то, что эта группа является ключевой для разделения веществ по кластерам.")
