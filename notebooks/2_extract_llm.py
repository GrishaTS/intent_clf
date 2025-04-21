import os

import pandas as pd
import requests
from tqdm import tqdm

# Конфигурация API
API_URL = "http://localhost:27361/generate/ollama"
model = "qwen2.5:7b-instruct-q4_0"

# Пути к файлам
input_path = "data/df.csv"
output_path = "data/df_llm.csv"

# Читаем исходный датасет
df = pd.read_csv(input_path)
df["description"] = df["description"].fillna("")

# Если выходной файл уже существует, подгружаем его и обновляем колонку "abstract"
if os.path.exists(output_path):
    df_out = pd.read_csv(output_path)
    if "abstract" in df_out.columns:
        df["abstract"] = df_out["abstract"]
    else:
        df["abstract"] = ""
else:
    df["abstract"] = ""

# Определяем строки, в которых абстракт отсутствует (пустая строка или только пробелы)
mask = df["abstract"].isnull() | (df["abstract"].str.strip() == "")
df_to_process = df[mask].copy()
print(f"Будет обработано {len(df_to_process)} строк из {len(df)}.")


def get_abstract(text):
    prompt = (
        "На основе следующего текста сформируй краткий абстракт. Верни только абстракт по содержимому без приветсвий и т.п. "
        f"Текст: {text}\nАбстракт (на русском, не более 300 символов):"
    )
    variables = {"text": text}
    try:
        response = requests.post(
            API_URL,
            json={
                "model": model,
                "stream": False,
                "prompt": prompt,
                "variables": variables,
            },
            timeout=60,
        )
        if response.status_code == 200:
            # Если API возвращает результат как строку
            return response.text.strip()
        else:
            print(f"Ошибка запроса: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Ошибка при вызове API: {e}")
        return ""


# Обработка батчами
batch_size = 10
num_batches = (len(df_to_process) + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Обработка батчей"):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(df_to_process))
    batch_indices = df_to_process.index[start:end]
    abstracts = []
    for idx in batch_indices:
        text = df.loc[idx, "description"]
        abstract = get_abstract(text)
        abstracts.append(abstract)
    # Обновляем исходный DataFrame по найденным индексам
    df.loc[batch_indices, "abstract"] = abstracts
    # Сохраняем промежуточный результат после каждого батча
    df.to_csv(output_path, index=False)
    tqdm.write(f"Батч {i + 1}/{num_batches} обработан и сохранён.")

print(f"Датасет с абстрактами сохранён в {output_path}")
