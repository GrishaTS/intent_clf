import time

from config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def wait_for_qdrant():
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    while True:
        try:
            client.get_collections()
            print("Qdrant is ready!")
            break
        except Exception as e:
            print(f"Waiting for Qdrant: {e}")
            time.sleep(1)


if __name__ == "__main__":
    wait_for_qdrant()
