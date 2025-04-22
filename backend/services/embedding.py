import logging
import time
from typing import List, Union

import numpy as np
import torch
from config import settings
from transformers import AutoModel, AutoTokenizer

# Настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedding_service")


class E5Embedder:
    def __init__(self):
        logger.info(f"Initializing E5Embedder with model: {settings.MODEL_NAME}")
        self.model_name = settings.MODEL_NAME
        self.max_length = settings.MAX_LENGTH
        self.batch_size = settings.BATCH_SIZE

        # Проверка доступности CUDA и настройка устройства
        if settings.USE_CUDA == "1":
            logger.info("CUDA requested. Checking availability...")
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(
                    f"Using GPU: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.2f} GB memory"
                )
            else:
                self.device = "cpu"
                logger.warning("CUDA requested but not available. Falling back to CPU.")
        else:
            self.device = "cpu"
            logger.info("Using CPU as specified in settings")

        self.tokenizer = None
        self.model = None
        logger.info(
            f"E5Embedder initialized. Device: {self.device}, Batch size: {self.batch_size}"
        )

    def load_model(self):
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading model {self.model_name}...")
            start_time = time.time()

            try:
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info("Tokenizer loaded successfully")

                logger.info("Loading model...")
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Moving model to device: {self.device}")
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded and set to evaluation mode")

                load_time = time.time() - start_time
                logger.info(f"Model loading completed in {load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
        else:
            logger.debug("Model already loaded, reusing existing instance")
        return self

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Извлекает эмбединги из текстов с использованием модели E5
        """
        logger.info(f"Extracting embeddings for {len(texts)} texts")
        self.load_model()

        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        try:
            with torch.no_grad():
                for i in range(0, len(texts), self.batch_size):
                    batch_start_time = time.time()
                    batch_texts = texts[i : i + self.batch_size]
                    batch_num = i // self.batch_size + 1

                    logger.info(
                        f"Processing batch {batch_num}/{total_batches} with {len(batch_texts)} texts"
                    )

                    logger.debug("Tokenizing batch...")
                    encoded = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    logger.debug(f"Moving tensors to {self.device}")
                    try:
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    except RuntimeError as e:
                        logger.error(f"Error moving tensors to device: {str(e)}")
                        if "CUDA" in str(e):
                            logger.error("CUDA error detected. Falling back to CPU")
                            self.device = "cpu"
                            self.model.to("cpu")
                            encoded = {k: v.to("cpu") for k, v in encoded.items()}

                    logger.debug("Running model inference...")
                    output = self.model(**encoded)
                    token_embeddings = output.last_hidden_state
                    attention_mask = encoded["attention_mask"]

                    logger.debug("Computing mean pooling...")
                    # Расширяем маску для перемножения с эмбеддингами
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, dim=1
                    )
                    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                    batch_emb = (sum_embeddings / sum_mask).cpu().numpy()

                    embeddings.append(batch_emb)

                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Batch {batch_num}/{total_batches} processed in {batch_time:.2f} seconds"
                    )

            total_time = time.time() - start_time
            logger.info(f"All embeddings extracted in {total_time:.2f} seconds")

            result = np.concatenate(embeddings, axis=0)
            logger.info(f"Final embeddings shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
            raise

    def get_combined_embeddings(
        self, subjects: List[str], descriptions: List[str]
    ) -> np.ndarray:
        """
        Извлекает и комбинирует эмбединги из тем и описаний
        """
        if len(subjects) != len(descriptions):
            logger.error(
                f"Mismatch in lengths: subjects={len(subjects)}, descriptions={len(descriptions)}"
            )
            raise ValueError("Subjects and descriptions must have the same length")

        logger.info(f"Extracting combined embeddings for {len(subjects)} items")

        logger.info("Extracting subject embeddings...")
        start_time = time.time()
        subject_embeddings = self.get_embeddings(subjects)
        subject_time = time.time() - start_time
        logger.info(
            f"Subject embeddings extracted in {subject_time:.2f} seconds. Shape: {subject_embeddings.shape}"
        )

        logger.info("Extracting description embeddings...")
        start_time = time.time()
        description_embeddings = self.get_embeddings(descriptions)
        desc_time = time.time() - start_time
        logger.info(
            f"Description embeddings extracted in {desc_time:.2f} seconds. Shape: {description_embeddings.shape}"
        )

        logger.info("Combining embeddings...")
        combined = np.concatenate([subject_embeddings, description_embeddings], axis=1)
        logger.info(f"Combined embeddings shape: {combined.shape}")

        return combined


# Создаем синглтон для переиспользования модели
logger.info("Creating E5Embedder singleton instance")
embedder = E5Embedder()
