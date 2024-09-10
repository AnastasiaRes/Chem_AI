import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt

# Путь к файлу относительно местоположения текущего скрипта
base_dir = os.path.dirname(__file__)  # Получение пути к директории, где находится текущий скрипт
file_path = os.path.join(base_dir, 'full_dataset.csv')  # Путь к исходному файлу
backup_file_path = os.path.join(base_dir, 'full_drug_backup.csv')  # Путь к резервной копии файла

# Создание копии исходного файла
if not os.path.exists(backup_file_path):  # Проверка, чтобы не перезаписать уже существующую копию
    data = pd.read_csv(file_path)
    data.to_csv(backup_file_path, index=False)  # Сохранение резервной копии без индексов
else:
    data = pd.read_csv(backup_file_path)  # Работа с уже существующей копией

# Разделение данных на числовые и категориальные колонки
numerical = data.select_dtypes(include=['int64', 'float64'])
categorical = data.select_dtypes(include=['object'])

print("Numerical columns:")
print(numerical.columns)
print("Categorical columns:")
print(categorical.columns)

# Функция для очистки данных и отслеживания удаленных значений
def clean_data(df):
    # Изначальные размеры
    initial_shape = df.shape

    # Удаление дубликатов
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - df_cleaned.shape[0]

    # Обработка пропущенных значений
    df_cleaned = df_cleaned.dropna(thresh=len(df.columns) * 0.5)  # Удаление строк с более чем 50% пропущенных значений
    rows_removed_na = initial_shape[0] - df_cleaned.shape[0] - duplicates_removed
    df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))  # Замена пропущенных значений медианой для числовых колонок

    # Обработка выбросов
    outliers_removed = 0
    outliers_info = []  # Список для хранения информации о выбросах
    for column in numerical.columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Поиск выбросов перед удалением
        outliers = df_cleaned[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)]
        if not outliers.empty:
            outliers_info.append((column, outliers[column].tolist()))  # Сохраняем колонку и значения выбросов

        # Удаление выбросов
        initial_count = df_cleaned.shape[0]
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        outliers_removed += (initial_count - df_cleaned.shape[0])

    return df_cleaned, duplicates_removed, rows_removed_na, outliers_removed, outliers_info

# Применение очистки к данным и отображение результатов очистки
cleaned_data, duplicates_removed, rows_removed_na, outliers_removed, outliers_info = clean_data(data)

# Печать информации о том, что было удалено
print(f"Удалено дубликатов: {duplicates_removed}")
print(f"Удалено строк с пропущенными значениями: {rows_removed_na}")
print(f"Удалено выбросов: {outliers_removed}")

# Печать информации о выбросах
if outliers_info:
    print("\nСписок выбросов по колонкам:")
    for column, outliers in outliers_info:
        print(f"Колонка: {column}, Количество выбросов: {len(outliers)}, Значения выбросов: {outliers}")
else:
    print("Выбросов не обнаружено.")

"""## Функция для построения violin plot по всем колонкам
def violin_plot(cleaned_data):
    num_columns = len(cleaned_data.columns)
    plots_per_row = 4  # Количество графиков в строке
    for i in range(0, num_columns, plots_per_row):
        end = min(i + plots_per_row, num_columns)  # Определяем конец диапазона
        fig, ax_ = plt.subplots(1, end - i, figsize=(12, 4))  # Создаем подграфики для каждого набора из 4 колонок
        ax = ax_.flatten() if (end - i) > 1 else [ax_]  # Убедитесь, что ax является списком даже для одного графика
        for number, column in enumerate(cleaned_data.columns[i:end]):  # Используем cleaned_data для визуализаций
            sns.violinplot(data=cleaned_data, x=column, ax=ax[number])
        fig.suptitle(f"Violin plots for numerical parameters ({i + 1} to {end})")
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.6)
        plt.show()

violin_plot(cleaned_data)"""

# Путь для сохранения очищенного файла
cleaned_file_path = os.path.join(base_dir, 'cleaned_dataset.csv')

# Сохранение очищенных данных в новый CSV файл
cleaned_data.to_csv(cleaned_file_path, index=False)  # Сохраняем данные без индексов

