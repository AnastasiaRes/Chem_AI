import pandas as pd
import os
import seaborn as sn

# Путь к файлу относительно местоположения текущего скрипта
base_dir = os.path.dirname(__file__)  # Получение пути к директории, где находится текущий скрипт
file_path = os.path.join(base_dir, 'psyche_drug')  # Построение полного пути к файлу, то есть к части пути base_dir добавляется 'psyche_drug'

# Чтение CSV файла
data = pd.read_csv(file_path)
print(data.head())
