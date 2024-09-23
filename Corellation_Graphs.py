# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:42:13 2024

@author: Adelina
"""

#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.feature_selection import VarianceThreshold
import scipy.stats as stats
from scipy.stats import shapiro
'''from scipy.stats import pearsonr
from scipy.stats import spearmanr
'''

#%%
# Получаем путь к директории, где находится скрипт
base_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу с данными
data_path = os.path.join(base_dir, 'full_dataset.csv')

# Загрузка датасета
df = pd.read_csv(data_path)
df_numeric = df.select_dtypes(exclude=['object']).copy()
print(df.columns)

#%%
# Хитмап всего датасета
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(20, 16))  
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

#%%
def variance_threshold(df_numeric,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df_numeric)
    new_cols = var_thres.get_support()
    return df_numeric.iloc[:,new_cols]

df_variance = variance_threshold(df_numeric, 0)

#%%
# Хитмап для очищенного датасета
plt.figure(figsize = (20, 16))
sns.heatmap(df_variance.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Reduced Dataset 2')
plt.show()

#%%
# Вычисление корреляционной матрицы
correlation_matrix = df_variance.corr().abs()
# Создание маски для верхнего треугольника корреляционной матрицы
upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
# Применение маски к корреляционной матрице
upper_corr_matrix = correlation_matrix.where(upper_triangle)
# Удаление признаков с высокой корреляцией
threshold = 0.9  
to_drop = [column for column in upper_corr_matrix.columns if any(upper_corr_matrix[column] > threshold)]
# Создание нового DataFrame без сильно коррелированных признаков
df_reduced = df_variance.drop(columns=to_drop)
# # print(f"Удаленные признаки: {to_drop}")

#%%
# Хитмап для очищенного очищенного датасета
plt.figure(figsize=(20, 16))
sns.heatmap(df_reduced.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Reduced Dataset')
plt.show()

#%%
is_psychedelic = 'is_psychedelic'
correlations = df_reduced.corr()[is_psychedelic].drop(is_psychedelic)

#%%
'''# Создаем точечные графики для каждой переменной
features = df_reduced.columns.drop(is_psychedelic)
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_reduced, x=feature, y=is_psychedelic)
    correlation, p_value = pearsonr(df_reduced[feature], df_reduced[is_psychedelic])
    plt.title(f'Корреляция Пирсона между {feature} и {is_psychedelic}')
    plt.xlabel(feature)
    plt.ylabel(is_psychedelic)
    plt.grid(True)
    plt.text(0.05, 0.5, f'Корреляция: {correlation:.2f}, p-значение: {p_value:.4f}', 
             horizontalalignment='left', 
             verticalalignment='center', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    
# Создаем точечные графики для каждой переменной по спирмену
correlations = df_reduced.corr(method='spearman')[is_psychedelic].drop(is_psychedelic)
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_reduced, x=feature, y=is_psychedelic, color='red')
    correlation, p_value = spearmanr(df_reduced[feature], df_reduced['is_psychedelic'])
    plt.title(f'Корреляция Спирмена между {feature} и {is_psychedelic}')
    plt.xlabel(feature)
    plt.ylabel(is_psychedelic)
    plt.grid(True)
    plt.text(0.05, 0.5, f'Корреляция: {correlation:.2f}, p-значение: {p_value:.4f}', 
             horizontalalignment='left', 
             verticalalignment='center', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()'''
    
#%%
# T test

group_psy = df_variance[df_variance['is_psychedelic'] == 1]
group_non_psy = df_variance[df_variance['is_psychedelic'] == 0]

features = df_variance.columns.tolist()
features.remove('is_psychedelic')

# Проверка на нормальность
normal_features = pd.DataFrame()

for feature in features:
    statistic_psy, p_value_psy = shapiro(group_psy[feature].values)
    statistic_non_psy, p_value_non_psy = shapiro(group_non_psy[feature])
    print(f"Тест Шапиро-Уилка для психоделиков для столбца '{feature}':  W={statistic_psy:.3f} p-value = {p_value_psy:.4f}")
    print(f"Тест Шапиро-Уилка для не психоделиков для столбца '{feature}':  W={statistic_non_psy:.3f} p-value = {p_value_non_psy:.4f}")
    if p_value_psy > 0.05 and p_value_non_psy > 0.05:
        normal_features[feature] = df[feature] 

normal_features['is_psychedelic'] = df['is_psychedelic']
print('Столбцы с нормальным распределением:')
print(normal_features)
                 
#%%
# Т test непосредственно
results = []

for feature in features:
     # Извлекаем значения для каждой группы
     psy_values = group_psy[feature].values
     non_psy_values = group_non_psy[feature].values
     t_stat, p_value = stats.ttest_ind(psy_values, non_psy_values, equal_var=False)
     results.append({
         'Feature': feature,
         't-statistic': t_stat,
         'p-value': p_value
         })
        
results_df = pd.DataFrame(results)
'''
# График распределения значений для каждой группы
def plot_distributions(feature, group_psy, group_non_psy):
    plt.figure(figsize=(20, 5))
    sns.histplot(group_psy[feature], kde=True, label='Psychedelic', color='blue', alpha=0.5)
    sns.histplot(group_non_psy[feature], kde=True, label='Non-Psychedelic', color='orange', alpha=0.5)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
'''
# График t-статистики и p-значения для каждого признака
def plot_t_test_results(results_df):
    plt.figure(figsize=(20, 10))
    sns.barplot(x='Feature', y='t-statistic', data=results_df, palette='viridis')
    plt.title('t-statistic for Each Feature')
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Feature', y='p-value', data=results_df, palette='viridis')
    plt.title('p-value for Each Feature')
    plt.xticks(rotation=90)
    plt.show()
    
'''
# Пример использования функций для построения графиков
for feature in features:
    plot_distributions(feature, group_psy, group_non_psy)'''

plot_t_test_results(results_df)

#%% тест проверки
'''
mu, sigma = 0, 1
sample1 = np.random.normal(0, 1, 1000)
sample2 = np.random.normal(1, 2, 1000)
sample3 = np.random.normal(0.5, 2, 1000)
samples = pd.DataFrame({
    'Sample1': sample1,
    'Sample2': sample2,
    'Sample3': sample3
})

for sample in samples:
    statistic, p_value = shapiro(samples[sample])
    print(f"Тест Шапиро-Уилка:  W={statistic:.3f} p-value = {p_value:.4f}")
    '''

#%%
# Отбор коррелирующих признаков
corr_matrix = df_reduced.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if 0.6 > any(upper[column] > 0.8)]
df_reduced.drop(to_drop, axis=1, inplace=True)

df_reduced['is_psychedelic'] = df['is_psychedelic']

sns.pairplot(df_variance, hue='is_psychedelic', palette={0: 'green', 1: 'violet'})
plt.show()

