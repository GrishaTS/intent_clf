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
    """Двухэтапный классификатор: сначала KNN в Qdrant, потом фильтрация reranker'ом."""

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
        """Считает reranker-оценки для пары (query, doc)."""
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
        self,
        user_id: str,
        subject: str,
        description: str,
        query_vector: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Предсказывает класс для запроса: top-K соседей + reranker."""
        query_text = f"{subject} {description}"

        # 1) Соседи из векторной БД (лимит=50)
        neighbors = vector_db.search_vectors(
            user_id=user_id,
            query_vector=query_vector,
            limit=50,
        )

        if not neighbors:
            logger.warning("No neighbors found in vector DB")
            return []

        neighbor_texts = [
            n.get("subject", "") + " " + n.get("description", "") for n in neighbors
        ]
        neighbor_labels = [n.get("class_name", "") for n in neighbors]
        sims = [float(n.get("score", 0.0)) for n in neighbors]

        # 2) Reranker
        scores = self._rerank_scores(query_text, neighbor_texts)

        # 3) Фильтрация по threshold
        filtered = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= self.threshold]

        # 4) Сортировка по убыванию reranker_score
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)

        # 5) Top-K
        topk = filtered_sorted[: self.k]
        kept_indices = [i for i, _ in topk]

        # 6) Fallback, если пусто
        if not kept_indices:
            sims_with_idx = list(enumerate(sims))
            sims_sorted = sorted(sims_with_idx, key=lambda x: x[1], reverse=True)
            kept_indices = [i for i, _ in sims_sorted[: self.k]]

        # 7) Агрегация
        def _aggregate(indices: List[int]) -> List[Dict[str, float]]:
            class_scores: Dict[str, float] = {}
            total_sim = 0.0
            for i in indices:
                lbl = neighbor_labels[i]
                sim = sims[i]
                class_scores[lbl] = class_scores.get(lbl, 0.0) + sim
                total_sim += sim

            if total_sim <= 1e-12:
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

        predictions = _aggregate(kept_indices)
        return predictions


# Singleton instance
classifier = FilteredKNNClassificationService()
