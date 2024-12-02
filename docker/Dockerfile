# Використовуємо офіційний Python-образ
FROM python:3.9-slim

# Встановлення додаткових залежностей
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    && apt-get clean

# Встановлення Python-залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Налаштування робочої директорії
WORKDIR /app

# Копіюємо проект у контейнер
COPY . .

# Визначаємо команду за замовчуванням
CMD ["python", "src/kmeans_clustering.py"]
