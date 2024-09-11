# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:42:13 2024

@author: Adelina
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt 

# Получаем путь к директории, где находится скрипт
base_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу с данными
data_path = os.path.join(base_dir, 'full_dataset.csv')

# Загрузка датасета
df = pd.read_csv(data_path)
'''print(df.head)'''

df.info()
print(df.columns)
num_rows = len(df)
print(f"Количество строк в датасете: {num_rows}")



