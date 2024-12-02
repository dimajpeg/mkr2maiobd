import pandas as pd
import numpy as np

# Функция для загрузки данных
def load_data(file_path):
    print("Завантаження даних...")
    return pd.read_excel(file_path)

# Функция для предварительной обработки данных
def preprocess_data(df):
    # Удаляем строки с пропущенными значениями в ключевых столбцах
    df = df.dropna(subset=['CustomerID', 'UnitPrice', 'Quantity'])

    # Добавляем случайные данные для возрастов (демонстрационные)
    df['Age'] = np.random.randint(18, 65, size=len(df))

    # Рассчитываем ежегодный доход (демонстрационные данные)
    df['Annual_Income'] = df['UnitPrice'] * df['Quantity']

    # Генерируем случайный Spending Score
    df['Spending_Score'] = np.random.randint(1, 100, size=len(df))

    # Оставляем только нужные столбцы
    df = df[['Age', 'Annual_Income', 'Spending_Score']]

    print("Пример данных после обработки:", df.head())
    return df

# Функция для сохранения обработанных данных
def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Дані збережено у файл: {output_path}")

# Основная функция
if __name__ == "__main__":
    input_path = 'data/raw/Online Retail.xlsx'
    output_path = 'data/processed/processed_data.csv'

    raw_data = load_data(input_path)
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data, output_path)
