import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from tabulate import tabulate


# Загрузка данных
def load_data(file_name):
    """
    Загружает данные из файла.
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, file_name)
    return pd.read_csv(file_path)

# Визуализация распределения по группам
def plot_histograms(df, columns, group_column):
    """
    Строит гистограммы распределения признаков для разных групп.
    """
    groups = {'Psychedelic': df[df[group_column] == 1],
              'Non-Psychedelic': df[df[group_column] == 0]}

    for column in columns:
        plt.figure(figsize=(12, 6))
        for group_name, group_data in groups.items():
            plt.hist(group_data[column].dropna(), bins=30, density=True, alpha=0.5,
                     label=group_name, edgecolor='black')

        plt.title(f'Distribution of {column} in Psychedelic vs Non-Psychedelic')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

# Применение PCA
def apply_pca(df, columns):
    """
    Применяет PCA для снижения размерности данных.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['is_psychedelic'] = df['is_psychedelic'].values
    return pca_df

# Применение t-SNE
def apply_tsne(df, columns, perplexity=30, n_iter=1000):
    """
    Применяет t-SNE для снижения размерности данных.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
    tsne_df['is_psychedelic'] = df['is_psychedelic'].values
    return tsne_df

# Визуализация результатов снижения размерности
def plot_dimensionality_reduction(reduction_df, title, x_label, y_label):
    """
    Строит график для визуализации снижения размерности.
    """
    plt.figure(figsize=(10, 6))
    for label, color in zip([1, 0], ['blue', 'orange']):
        plt.scatter(reduction_df[reduction_df['is_psychedelic'] == label][x_label],
                    reduction_df[reduction_df['is_psychedelic'] == label][y_label],
                    alpha=0.6, label='Psychedelic' if label else 'Non-Psychedelic',
                    edgecolor='black')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

# Определение оптимального количества кластеров
def find_optimal_clusters(X, max_clusters=10):
    """
    Определяет оптимальное количество кластеров с помощью метода локтя и коэффициента силуэта.
    """
    silhouette_scores = []
    inertia = []
    n_clusters_list = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
        inertia.append(kmeans.inertia_)
        n_clusters_list.append(n_clusters)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_list, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_list, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()


# Сравнение нескольких разных методов кластеризации в поисках оптимального
def compare_clustering_methods(tsne_df, optimal_clusters=4, eps=0.5, min_samples=5):
    X = tsne_df[['t-SNE1', 't-SNE2']].values

    # Инициализация словарей для хранения метрик
    metrics = {
        'KMeans': {},
        'DBSCAN': {},
        'Agglomerative': {}
    }

    # KMeans
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    tsne_df['KMeans_Cluster'] = kmeans.fit_predict(X)
    labels_kmeans = tsne_df['KMeans_Cluster']
    n_clusters_kmeans = len(set(labels_kmeans))

    # Проверяем, что количество кластеров больше 1 для корректного вычисления метрик
    if n_clusters_kmeans > 1 and n_clusters_kmeans < len(X):
        metrics['KMeans']['Silhouette'] = silhouette_score(X, labels_kmeans)
        metrics['KMeans']['Calinski-Harabasz'] = calinski_harabasz_score(X, labels_kmeans)
        metrics['KMeans']['Davies-Bouldin'] = davies_bouldin_score(X, labels_kmeans)
    else:
        metrics['KMeans']['Silhouette'] = None
        metrics['KMeans']['Calinski-Harabasz'] = None
        metrics['KMeans']['Davies-Bouldin'] = None

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    tsne_df['DBSCAN_Cluster'] = dbscan.fit_predict(X)
    labels_dbscan = tsne_df['DBSCAN_Cluster']
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)  # Исключаем шум (-1)

    if n_clusters_dbscan > 1:
        metrics['DBSCAN']['Silhouette'] = silhouette_score(X, labels_dbscan)
        metrics['DBSCAN']['Calinski-Harabasz'] = calinski_harabasz_score(X, labels_dbscan)
        metrics['DBSCAN']['Davies-Bouldin'] = davies_bouldin_score(X, labels_dbscan)
    else:
        metrics['DBSCAN']['Silhouette'] = None
        metrics['DBSCAN']['Calinski-Harabasz'] = None
        metrics['DBSCAN']['Davies-Bouldin'] = None

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=optimal_clusters)
    tsne_df['Agglomerative_Cluster'] = agglo.fit_predict(X)
    labels_agglo = tsne_df['Agglomerative_Cluster']
    n_clusters_agglo = len(set(labels_agglo))

    if n_clusters_agglo > 1 and n_clusters_agglo < len(X):
        metrics['Agglomerative']['Silhouette'] = silhouette_score(X, labels_agglo)
        metrics['Agglomerative']['Calinski-Harabasz'] = calinski_harabasz_score(X, labels_agglo)
        metrics['Agglomerative']['Davies-Bouldin'] = davies_bouldin_score(X, labels_agglo)
    else:
        metrics['Agglomerative']['Silhouette'] = None
        metrics['Agglomerative']['Calinski-Harabasz'] = None
        metrics['Agglomerative']['Davies-Bouldin'] = None

    # Вывод результатов
    for method in metrics:
        print(f"\nМетод кластеризации: {method}")
        cluster_column = f"{method}_Cluster"
        if cluster_column in tsne_df.columns and metrics[method]['Silhouette'] is not None:
            print(f"Количество кластеров: {len(set(tsne_df[cluster_column]))}")
            print(f"Silhouette Score: {metrics[method]['Silhouette']:.4f}")
            print(f"Calinski-Harabasz Index: {metrics[method]['Calinski-Harabasz']:.4f}")
            print(f"Davies-Bouldin Index: {metrics[method]['Davies-Bouldin']:.4f}")
        else:
            print("Недостаточно кластеров для вычисления метрик.")


# Кластеризация данных и визуализация (при сравнении выиграл KMeans)
def perform_clustering(tsne_df, optimal_clusters=4):
    """
    Выполняет кластеризацию данных и визуализирует результаты.
    """
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    tsne_df['Cluster'] = kmeans.fit_predict(tsne_df[['t-SNE1', 't-SNE2']])
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=tsne_df['Cluster'], cmap='viridis',
                alpha=0.6, edgecolor='black')
    plt.title('t-SNE Clustering with K-means')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()
    return tsne_df


def analyze_clusters_and_features(tsne_df, df, columns):
    """
    Анализирует кластеры и оценивает важность признаков различными методами.
    """
    # Объединение данных с информацией о кластерах
    df_with_clusters = df.reset_index(drop=True).join(tsne_df[['Cluster']].reset_index(drop=True))

    # Средние значения функциональных групп для каждого кластера
    cluster_means = df_with_clusters.groupby('Cluster')[columns].mean()

    # Количество психоделических и непсиходелических веществ в каждом кластере
    cluster_counts = df_with_clusters.groupby('Cluster')['is_psychedelic'].value_counts().unstack().fillna(0)
    cluster_counts.columns = ['Non-Psychedelic', 'Psychedelic']

    # Объединение данных о средних значениях и распределении веществ
    cluster_summary = pd.concat([cluster_means, cluster_counts], axis=1)
    print("=== Cluster Summary ===")
    print(cluster_summary)

    # Визуализация распределения по кластерам
    cluster_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Distribution of Psychedelic vs Non-Psychedelic Substances in Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Substance Type')
    plt.show()

    # Фильтрация значимых кластеров с преобладанием психоделических веществ
    cluster_summary['Total'] = cluster_summary['Psychedelic'] + cluster_summary['Non-Psychedelic']
    cluster_summary['Psychedelic_Ratio'] = cluster_summary['Psychedelic'] / cluster_summary['Total']
    significant_clusters = cluster_summary[cluster_summary['Psychedelic_Ratio'] > 0.6]
    print("\n=== Значимые кластеры с преобладанием психоделических веществ ===")
    print(significant_clusters)

    # Анализируем только значимые кластеры
    significant_df = df_with_clusters[df_with_clusters['Cluster'].isin(significant_clusters.index)]

    # Оценка важности признаков с помощью Random Forest на значимых кластерах
    X_significant = significant_df[columns]
    y_significant = significant_df['is_psychedelic']

    # Разделение данных на обучающую и тестовую выборки
    X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(
        X_significant, y_significant, test_size=0.3, random_state=42
    )

    # Обучение модели Random Forest
    rf_model_sig = RandomForestClassifier(random_state=42)
    rf_model_sig.fit(X_train_sig, y_train_sig)

    # Оценка важности признаков на значимых кластерах
    rf_importances_sig = pd.DataFrame(
        rf_model_sig.feature_importances_, index=columns, columns=['RandomForest_Importance_Significant']
    ).sort_values(by='RandomForest_Importance_Significant', ascending=False)

    # Оценка модели на тестовых данных
    y_pred_sig = rf_model_sig.predict(X_test_sig)
    print("\n=== Отчет по классификации Random Forest на значимых кластерах ===")
    print(classification_report(y_test_sig, y_pred_sig))
    print(f"Точность модели на значимых кластерах: {accuracy_score(y_test_sig, y_pred_sig):.2f}")

    # --- Проведение ANOVA анализа по кластерам ---
    print("\n=== ANOVA анализ по кластерам ===")
    anova_results_cluster = []
    for column in columns:
        groups = [
            df_with_clusters[df_with_clusters['Cluster'] == cluster][column]
            for cluster in df_with_clusters['Cluster'].unique()
        ]
        f_val, p_val = f_oneway(*groups)
        anova_results_cluster.append({'Feature': column, 'F-Value': f_val, 'P-Value': p_val})
    anova_df_cluster = pd.DataFrame(anova_results_cluster).set_index('Feature').sort_values(by='P-Value')
    print(anova_df_cluster)

    # Выполнение корреляционного анализа
    correlation_methods = ['pearson', 'spearman']
    correlation_results = {}
    for method in correlation_methods:
        # Вычисляем корреляцию между признаками и 'Cluster'
        corr = df_with_clusters[columns + ['Cluster']].corr(method=method)
        # Извлекаем корреляции признаков с 'Cluster'
        corr_with_cluster = corr['Cluster'].drop('Cluster')
        correlation_results[method.capitalize()] = corr_with_cluster
    correlation_df = pd.DataFrame(correlation_results)

    # Объединение результатов в сводную таблицу
    summary_df = rf_importances_sig.join(anova_df_cluster[['F-Value', 'P-Value']], how='outer')
    summary_df = summary_df.join(correlation_df, how='outer')
    summary_df = summary_df.sort_values(by='RandomForest_Importance_Significant', ascending=False)

    print("\n=== Сводная таблица результатов (Significant Clusters) ===")
    print(tabulate(summary_df, headers='keys', tablefmt='psql'))

    # Визуализация важности признаков по Random Forest на значимых кластерах
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RandomForest_Importance_Significant', y=summary_df.index, data=summary_df)
    plt.title('Feature Importances by Random Forest (Significant Clusters)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    # Визуализация корреляции признаков с номерами кластеров в виде тепловой карты
    def visualize_correlation_heatmap(correlation_df):
        """
        Визуализирует корреляцию признаков с номерами кластеров в виде тепловой карты.
        """
        # Транспонируем DataFrame для удобства отображения
        corr_matrix = correlation_df.T

        plt.figure(figsize=(16, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 8})
        plt.title('Correlation of Features with Cluster Number', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Correlation Method', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    # Визуализация тепловой карты
    visualize_correlation_heatmap(correlation_df)

# Новая функция для вычисления важности признаков для всех выбранных колнок
def calculate_feature_importance_full_func_dataset(df, columns):
    """
    Вычисляет важность признаков с помощью Random Forest на выбранном наборе данных.
    """
    X = df[columns]
    y = df['is_psychedelic']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучение модели Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Вычисление важности признаков
    rf_importances = pd.DataFrame(
        rf_model.feature_importances_, index=columns, columns=['RandomForest_Importance']
    ).sort_values(by='RandomForest_Importance', ascending=False)

    # Оценка модели на тестовых данных
    y_pred = rf_model.predict(X_test)
    print("\n=== Отчет по классификации Random Forest на выбранном наборе данных ===")
    print(classification_report(y_test, y_pred))
    print(f"Точность модели на выбранном наборе данных: {accuracy_score(y_test, y_pred):.2f}")

    return rf_importances


# Основной код выполнения
if __name__ == "__main__":
    df = load_data('full_dataset.csv')
    functional_group_columns = [
        'NumLipinskiHBA', 'NumLipinskiHBD', 'NumAmideBonds', 'NumHBD', 'NumHBA',
        'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
        'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
        'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles'
    ]


    # Обработка пропущенных значений
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("Пропущенные значения:")
        print(missing_values)
        df = df.dropna()
    else:
        print("Нет пропущенных значений.")


    # Визуализация распределения признаков
    # plot_histograms(df, functional_group_columns, 'is_psychedelic')


    # Применяем PCA и t-SNE
    pca_df = apply_pca(df, functional_group_columns)
    plot_dimensionality_reduction(pca_df, 'PCA of Functional Groups: Psychedelic vs Non-Psychedelic', 'PCA1', 'PCA2')

    tsne_df = apply_tsne(df, functional_group_columns)
    plot_dimensionality_reduction(tsne_df, 't-SNE of Functional Groups: Psychedelic vs Non-Psychedelic', 't-SNE1',
                                  't-SNE2')

    # Определяем оптимальное количество кластеров и выполняем кластеризацию
    find_optimal_clusters(tsne_df[['t-SNE1', 't-SNE2']], max_clusters=10)
    tsne_df = perform_clustering(tsne_df)

    # Сравниваем методы кластеризации
    compare_clustering_methods(tsne_df, optimal_clusters=4, eps=0.3, min_samples=5)


    # Анализ кластеров и оценка признаков
    analyze_clusters_and_features(tsne_df, df, functional_group_columns)

    # Вычисление важности признаков для всех выбранных колнок
    rf_importances_full = calculate_feature_importance_full_func_dataset(df, functional_group_columns)

    # Вывод результатов по Random Forest всех выбранных колнок
    print("\n=== Важность признаков по Random Forest для всех выбранных колнок ===")
    print(rf_importances_full)

    # Визуализация важности признаков
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RandomForest_Importance', y=rf_importances_full.index, data=rf_importances_full)
    plt.title('Feature Importances by Random Forest (Functional Groups Dataset)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()