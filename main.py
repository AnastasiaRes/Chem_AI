import argparse
import random
import pandas as pd
from utils import get_molecule_data, calculate_mqn, calculate_properties, calculate_vsa_descriptors, recalculate_slogp_for_missing

def generate_molecules(num_molecules, random_seed):
    """
    Генерация молекул на основе случайных CID из PubChem.

    Parameters:
    num_molecules (int): Количество молекул для генерации
    random_seed (int): Фиксированное значение seed для воспроизводимости

    Returns:
    pd.DataFrame: DataFrame с рассчитанными свойствами молекул
    """
    random.seed(random_seed)
    random_cids = random.sample(range(1, 1000000), num_molecules)

    # Получаем базовые данные о молекулах с помощью PubChem
    data = [get_molecule_data(cid) for cid in random_cids]
    
    # Преобразуем список данных в DataFrame
    df = pd.DataFrame(data)

    # Рассчитываем свойства и добавляем их в DataFrame
    df_mqn = df['Molecule (RDKit Mol)'].apply(lambda x: pd.Series(calculate_mqn(x)))
    df_properties = df['Molecule (RDKit Mol)'].apply(lambda x: pd.Series(calculate_properties(x)))
    df_vsa = df['Molecule (RDKit Mol)'].apply(lambda x: pd.Series(calculate_vsa_descriptors(x)))

    # Объединяем все свойства
    df_combined = pd.concat([df, df_mqn, df_properties, df_vsa], axis=1)

    # Перерасчет SlogP для None значений
    df_combined['SlogP'] = df_combined.apply(
        lambda row: recalculate_slogp_for_missing(row['Molecule (RDKit Mol)'], row['SlogP']), axis=1
    )

    return df_combined

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генерация базы данных непсиходелических молекул")
    parser.add_argument("--random_seed", type=int, help="Random seed for generation")
    parser.add_argument("--num_molecules", type=int, help="Number of molecules to generate")
    args = parser.parse_args()

    # Если аргументы командной строки не заданы, запрашиваем ввод с клавиатуры
    if args.num_molecules is None:
        try:
            num_molecules = int(input("Введите количество молекул (по умолчанию 700): ") or 700)
        except ValueError:
            print("Ошибка ввода! Используется значение по умолчанию (700).")
            num_molecules = 700
    else:
        num_molecules = args.num_molecules

    if args.random_seed is None:
        try:
            random_seed = int(input("Введите random seed (по умолчанию 42): ") or 42)
        except ValueError:
            print("Ошибка ввода! Используется значение по умолчанию (42).")
            random_seed = 42
    else:
        random_seed = args.random_seed

    # Генерация молекул
    df_molecules = generate_molecules(num_molecules, random_seed)

    # Сохранение результата
    df_molecules.to_csv("non_psychedelic_molecules.csv", index=False)
    print(f"Генерация завершена. Сохранено {num_molecules} молекул в 'non_psychedelic_molecules.csv'.")

if __name__ == "__main__":
    main()