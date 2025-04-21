import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("vector_db_service")


class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = 1024 * 2  # 1024 для subject + 1024 для description
        self.init_collection()
        logger.info(
            f"Initialized VectorDBService with host={settings.QDRANT_HOST}, port={settings.QDRANT_PORT}"
        )

    def init_collection(self):
        """
        Инициализирует коллекцию в Qdrant, если она еще не существует
        """
        logger.info(f"Checking if collection {self.collection_name} exists")
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            logger.info(
                f"Creating collection {self.collection_name} with vector size {self.vector_size}"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
                # Добавляем схему для payload, чтобы обеспечить правильную индексацию
                schema=models.CollectionSchema(
                    payload_schema={
                        "request_id": models.PayloadSchemaType.KEYWORD,
                        "subject": models.PayloadSchemaType.TEXT,
                        "description": models.PayloadSchemaType.TEXT,
                        "class_name": models.PayloadSchemaType.KEYWORD,
                        "task": models.PayloadSchemaType.KEYWORD,
                    }
                ),
            )

            # Создаем индекс для поля request_id
            self.client.create_field_index(
                collection_name=self.collection_name,
                field_name="request_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Collection {self.collection_name} created successfully")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def string_to_uuid(self, string_id):
        """
        Преобразует строковый ID в UUID версии 5
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    def upload_vectors(
        self, vectors: np.ndarray, payloads: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Загружает векторы и метаданные в Qdrant.
        Требует, чтобы в каждом элементе payloads был ключ 'request_id'.
        """
        # Проверяем, что у всех элементов есть request_id
        missing_ids = [i for i, p in enumerate(payloads) if "request_id" not in p]
        if missing_ids:
            raise ValueError(
                f"Missing 'request_id' in payloads at indices: {missing_ids}"
            )

        # Создаем точки для загрузки
        points = []
        ids = []

        for i in range(len(vectors)):
            # Генерируем UUID на основе строкового ID
            original_id = payloads[i]["request_id"]
            point_id = self.string_to_uuid(original_id)
            ids.append(point_id)

            points.append(
                models.PointStruct(
                    id=point_id, vector=vectors[i].tolist(), payload=payloads[i]
                )
            )

        logger.info(
            f"Uploading {len(vectors)} vectors to Qdrant collection {self.collection_name}"
        )

        # Загружаем векторы в Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Successfully uploaded {len(vectors)} vectors")
        return ids

    def search_vectors(
        self, query_vector: np.ndarray, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Ищет ближайшие векторы в Qdrant
        """
        logger.info(
            f"Searching for {limit} closest vectors in collection {self.collection_name}"
        )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
        )

        results = []
        for res in search_result:
            result = {
                "id": res.id,
                "request_id": res.payload.get("request_id", ""),
                "score": res.score,
                **res.payload,
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")
        return results

    def search_by_id(self, request_id: str) -> Dict[str, Any]:
        """
        Ищет запись по оригинальному ID
        """
        logger.info(f"Searching for record with request_id={request_id}")

        search_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="request_id", match=models.MatchValue(value=request_id)
                    )
                ]
            ),
            limit=1,
        )

        if search_result[0]:  # Если есть результаты
            point = search_result[0][0]
            result = {
                "id": point.id,
                "request_id": point.payload.get("request_id", ""),
                **point.payload,
            }
            return result

        logger.warning(f"No record found with request_id={request_id}")
        return None


# Создаем синглтон для переиспользования клиента Qdrant
vector_db = VectorDBService()
