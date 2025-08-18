import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings
from services.vector_db import vector_db

logger = logging.getLogger("classification_service")


class FilteredKNNClassificationService:
    """Two-stage classifier using KNN search in Qdrant and reranker filtering."""

    def __init__(self) -> None:
        self.k = settings.KNN_NEIGHBORS
        self.threshold = settings.RERANK_THRESHOLD
        self.device = settings.DEVICE

        logger.info(
            "Loading reranker model %s on %s", settings.RERANKER_MODEL, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL)
            .to(self.device)
        )
        self.model.eval()

    def _rerank_scores(self, query: str, docs: List[str]) -> List[float]:
        """Computes reranker scores for a query and list of documents."""
        pairs = [(query, doc) for doc in docs]
        encoded = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits.cpu().numpy()

        if logits.shape[1] == 2:
            exp = np.exp(logits)
            probs = exp[:, 1] / exp.sum(axis=1)
            return probs.tolist()
        return logits[:, 0].tolist()

    def predict(
        self, subject: str, description: str, query_vector: np.ndarray, collection_name: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """Return class probabilities для запроса с учётом топ‐K после реранжирования."""
        query_text = f"{subject} {description}"

        # 1) Получаем соседей из векторной БД (лимит=50)
        neighbors = vector_db.search_vectors(query_vector, limit=50,)  # collection_name=collection_name

        if not neighbors:
            logger.warning("No neighbors found in vector DB")
            return []

        neighbor_texts = [
            n.get("subject", "") + " " + n.get("description", "") for n in neighbors
        ]
        neighbor_labels = [n.get("class_name", "") for n in neighbors]
        sims = [float(n.get("score", 0.0)) for n in neighbors]

        # 2) Считаем reranker‐оценки (пробабилистыческие, от 0 до 1)
        scores = self._rerank_scores(query_text, neighbor_texts)

        # 3) Оставляем только те индексы, у которых reranker_score >= threshold
        filtered = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= self.threshold]

        # 4) Сортируем их по убыванию reranker_score
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)

        # 5) Берём не более self.k самых релевантных
        topk = filtered_sorted[: self.k]
        kept_indices = [i for i, _ in topk]

        # 6) Если после фильтрации и топ‐K ничего не осталось, делаем fallback:
        if not kept_indices:
            # fallback: можно взять просто top-K по изначальной sim из vector DB
            # Например, сортируем neighbors по sims и берём первые k
            sims_with_idx = list(enumerate(sims))
            sims_sorted = sorted(sims_with_idx, key=lambda x: x[1], reverse=True)
            kept_indices = [i for i, _ in sims_sorted[: self.k]]

        # Вспомогательная функция для агрегации
        def _aggregate(indices: List[int]) -> List[Dict[str, float]]:
            class_scores: Dict[str, float] = {}
            total_sim = 0.0
            for i in indices:
                lbl = neighbor_labels[i]
                sim = sims[i]
                class_scores[lbl] = class_scores.get(lbl, 0.0) + sim
                total_sim += sim

            if total_sim <= 1e-12:
                # если сумма всех sim близка к 0, тогда считаем равномерные вероятности по частоте
                counts = Counter([neighbor_labels[i] for i in indices])
                total = sum(counts.values())
                return [
                    {"class_name": lbl, "probability": cnt / total}
                    for lbl, cnt in counts.most_common()
                ]

            return [
                {"class_name": lbl, "probability": score / total_sim}
                for lbl, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            ]

        # 7) Собираем предсказание по отфильтрованным индексам
        predictions = _aggregate(kept_indices)
        return predictions


# Singleton instance
classifier = FilteredKNNClassificationService()
