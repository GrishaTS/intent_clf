from typing import Optional, Literal
from ctx import Ctx
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

from api_utils import filter_high_quality_classes, get_token, upload_data, compute_metrics_backend, clear_index

BACKEND_API_URL = os.getenv("BACKEND_API_URL")


def retrain(ctx: Ctx) -> None:
    # Необходимые переменные от апи ctx.data_api_url
    data = pd.read_csv("test_data/test.csv")
    id_col = "Id"
    subject_col: Optional[str] = None
    description_col = "Body"
    class_col = "Target"
    filter_method: Literal["by_freq", "by_quality"] = "by_freq"
    top_n_values: int = 3           # для by_freq
    min_samples: int = 10           # для by_min_samples
    min_f1_score: float = 0.5       # для by_min_samples
    clear_index_flag: bool = True   # True - переобучние; False - дообучение
    username: str = "admin"
    password: str = "secret"
    
    #################################
    
    n = random.randint(200, 400)
    data = data.head(n)

    data = data.dropna(subset=[class_col])
    base_cols = [id_col, description_col, class_col]
    data_processed = data[base_cols].rename(
        columns={id_col: "id", description_col: "description", class_col: "class"}
    )
    if subject_col:
        data_processed["subject"] = data[subject_col]
    else:
        data_processed["subject"] = "no_subject"
    data_processed = data_processed[["id", "subject", "description", "class"]]
    
    if filter_method == "by_freq":
        top_values = data_processed["class"].value_counts().head(top_n_values).index.tolist()
        data_processed["class"] = data_processed["class"].apply(
            lambda x: x if x in top_values else "Others"
        )
    elif filter_method == "by_quality":
        data_high_quality, _ = filter_high_quality_classes(
            data_processed,
            min_samples=min_samples,
            min_f1_score=min_f1_score,
            api_url=BACKEND_API_URL,
            username=username,
            password=password,
        )
        if data_high_quality is not None:
            data_processed = data_high_quality
    else:
        return
    train_df, test_df = train_test_split(
        data_processed, test_size=0.1, random_state=42, stratify=data_processed["class"]
    )

    token = get_token(BACKEND_API_URL, username, password)
    if not token:
        return
    if clear_index_flag:
        clear_index(token, BACKEND_API_URL)
    if not upload_data(train_df, token, BACKEND_API_URL):
        return

    compute_metrics_backend(test_df, token, BACKEND_API_URL)

    upload_data(test_df, token, BACKEND_API_URL)
