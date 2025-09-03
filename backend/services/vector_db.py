import logging
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
from config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("vector_db_service")


def _user_collection_name(user_id):
    return f"u_{user_id}"


class VectorDBService:
    """
    Работа с Qdrant в парадигме: у каждого пользователя своя коллекция.
    Все методы принимают user_id и действуют в пределах его коллекции.
    """

    def __init__(self, vector_size: int = 1024 * 2):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.vector_size = vector_size
        self._known_collections = set()  # локальный кэш существующих коллекций
        logger.info(
            f"Initialized VectorDBService (multi-collection) host={settings.QDRANT_HOST}, port={settings.QDRANT_PORT}"
        )

    # ===== infra =====
    def _ensure_user_collection(self, user_id: Union[str, uuid.UUID]):
        """
        Создаёт коллекцию пользователя при отсутствии + индексы по payload.
        """
        name = _user_collection_name(user_id)
        if name in self._known_collections:
            return name

        existing = {c.name for c in self.client.get_collections().collections}
        if name not in existing:
            logger.info(f"Creating collection {name} (size={self.vector_size})")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
                shard_number=1,
                replication_factor=1,
                on_disk_payload=False,
            )

            # Индексы для типовых полей
            for field_name, field_type in {
                "request_id":  models.PayloadSchemaType.KEYWORD,
                "subject":     models.PayloadSchemaType.TEXT,
                "description": models.PayloadSchemaType.TEXT,
                "class_name":  models.PayloadSchemaType.KEYWORD,
                "task":        models.PayloadSchemaType.KEYWORD,
            }.items():
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type,
                )

            logger.info(f"Collection {name} created")
        else:
            logger.info(f"Collection {name} already exists")

        self._known_collections.add(name)
        return name

    # ===== utils =====
    def string_to_uuid(self, string_id: str) -> str:
        """Детерминированный UUIDv5 из любой строки (для point.id)."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    # ===== API =====
    def upload_vectors(
        self,
        user_id: Union[str, uuid.UUID],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Загрузка векторов в коллекцию пользователя.
        Требуется payload['request_id'] для каждого элемента.
        Возвращает список point.id (UUIDv5 от request_id).
        """
        coll = self._ensure_user_collection(user_id)

        missing = [i for i, p in enumerate(payloads) if "request_id" not in p]
        if missing:
            raise ValueError(f"Missing 'request_id' in payloads at indices: {missing}")

        if len(vectors) != len(payloads):
            raise ValueError("vectors и payloads должны быть одинаковой длины")

        points, ids = [], []
        for vec, pl in zip(vectors, payloads):
            pid = self.string_to_uuid(pl["request_id"])
            ids.append(pid)
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=pl,
                )
            )

        logger.info(f"[{coll}] upsert {len(points)} points")
        self.client.upsert(collection_name=coll, points=points)
        return ids

    def search_vectors(
        self,
        user_id: Union[str, uuid.UUID],
        query_vector: np.ndarray,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Поиск ближайших векторов в пределах коллекции пользователя.
        """
        coll = self._ensure_user_collection(user_id)

        logger.info(f"[{coll}] search top-{limit}")
        hits = self.client.search(
            collection_name=coll,
            query_vector=query_vector.tolist(),
            limit=limit,
        )

        results = []
        for h in hits:
            results.append(
                {
                    "id": h.id,
                    "request_id": h.payload.get("request_id", ""),
                    "score": h.score,
                    **h.payload,
                }
            )
        return results

    def search_by_id(
        self,
        user_id: Union[str, uuid.UUID],
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск записи по оригинальному request_id внутри коллекции пользователя.
        """
        coll = self._ensure_user_collection(user_id)

        logger.info(f"[{coll}] scroll by request_id={request_id}")
        found, _ = self.client.scroll(
            collection_name=coll,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="request_id",
                        match=models.MatchValue(value=request_id),
                    )
                ]
            ),
            limit=1,
        )

        if found:
            p = found[0]
            return {
                "id": p.id,
                "request_id": p.payload.get("request_id", ""),
                **p.payload,
            }
        return None

    def clear_user_collection(self, user_id: Union[str, uuid.UUID]) -> bool:
        """
        Полностью очищает коллекцию конкретного пользователя (drop + recreate).
        """
        coll = _user_collection_name(user_id)
        logger.info(f"[{coll}] drop collection")
        try:
            self.client.delete_collection(collection_name=coll)
            # удалить из кэша, при следующем вызове пересоздастся
            self._known_collections.discard(coll)
            # ленивое пересоздание при следующем обращении
            return True
        except Exception as e:
            logger.error(f"[{coll}] drop error: {e}")
            raise


# Синглтон
vector_db = VectorDBService()
