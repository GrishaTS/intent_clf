# embedding.py
import logging
import time
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedding_service")


def _to_device(batch: dict, device: torch.device, non_blocking: bool) -> dict:
    return {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class E5Embedder:
    """
    Лёгкий и надёжный экстрактор эмбеддингов.
    - Без mixed device: модель и входы всегда на одном девайсе.
    - Без device_map='auto'.
    - Возвращает np.ndarray(dtype=float32).
    """

    def __init__(self) -> None:
        self.model_name: str = settings.MODEL_NAME
        self.max_length: int = int(getattr(settings, "MAX_LENGTH", 256))
        self.batch_size: int = int(getattr(settings, "BATCH_SIZE", 32))
        self.use_cuda_flag: bool = str(getattr(settings, "USE_CUDA", "0")) == "1"
        self.use_amp: bool = bool(getattr(settings, "USE_AMP", True))  # AMP только на CUDA

        self.device: torch.device = self._select_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None

        logger.info(
            f"E5Embedder init: model='{self.model_name}', "
            f"device='{self.device}', batch_size={self.batch_size}, "
            f"max_length={self.max_length}, amp={self.use_amp}"
        )

    def _select_device(self) -> torch.device:
        if self.use_cuda_flag and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"Using CUDA: {props.name}, {props.total_memory / 1024**3:.2f} GB")
            return torch.device("cuda:0")
        if self.use_cuda_flag and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available → CPU fallback.")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        t0 = time.time()
        logger.info(f"Loading HF tokenizer/model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded in {time.time() - t0:.2f}s")

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, T, H); attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-9)                         # (B, 1)
        return summed / denom                                           # (B, H)

    def _encode_batch(self, texts: List[str]) -> dict:
        # padding='longest' экономит память; truncation с max_length — обязательны
        return self.tokenizer(
            texts,
            padding=True,              # к самому длинному в батче
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _forward_once(self, encoded: dict) -> torch.Tensor:
        non_blocking = self.device.type == "cuda"
        encoded = _to_device(encoded, self.device, non_blocking=non_blocking)

        amp_enabled = self.use_amp and self.device.type == "cuda"
        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(**encoded)
            else:
                out = self.model(**encoded)

        pooled = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
        return pooled  # (B, H) on self.device

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Возвращает эмбеддинги размера (N, H) как np.float32.
        """
        self._ensure_loaded()

        if not texts:
            h = int(self.model.config.hidden_size)  # type: ignore
            return np.empty((0, h), dtype=np.float32)

        outputs: List[torch.Tensor] = []
        total = len(texts)
        total_batches = (total + self.batch_size - 1) // self.batch_size

        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            batch = texts[s:e]
            b_idx = s // self.batch_size + 1
            logger.info(f"Embedding batch {b_idx}/{total_batches} (size={len(batch)})")

            encoded = self._encode_batch(batch)

            # 1-я попытка на текущем девайсе
            try:
                pooled = self._forward_once(encoded)
            except RuntimeError as ex:
                # Одноразовый откат на CPU при CUDA-ошибке
                if self.device.type == "cuda":
                    msg = str(ex).lower()
                    if "cuda" in msg or "device-side assert" in msg or "cublas" in msg:
                        logger.warning(f"CUDA failure on batch {b_idx}: {ex}. Fallback to CPU for this run.")
                        self.device = torch.device("cpu")
                        assert self.model is not None
                        self.model.to(self.device)
                        pooled = self._forward_once(encoded)
                    else:
                        raise
                else:
                    raise

            outputs.append(pooled.detach().cpu())

        embs = torch.cat(outputs, dim=0).contiguous().to(dtype=torch.float32).numpy()
        logger.info(f"Embeddings ready: shape={embs.shape}, dtype={embs.dtype}")
        return embs

    def get_combined_embeddings(self, subjects: List[str], descriptions: List[str]) -> np.ndarray:
        """
        Конкатенация эмбеддингов subject и description → (N, 2H), np.float32.
        """
        if len(subjects) != len(descriptions):
            raise ValueError(f"Lengths mismatch: subjects={len(subjects)}, descriptions={len(descriptions)}")

        logger.info(f"Extracting combined embeddings for N={len(subjects)}")
        subj = self.get_embeddings(subjects)        # (N, H)
        desc = self.get_embeddings(descriptions)    # (N, H)
        combined = np.concatenate([subj, desc], axis=1).astype(np.float32, copy=False)
        logger.info(f"Combined shape={combined.shape}, dtype={combined.dtype}")
        return combined


# Синглтон
logger.info("Creating E5Embedder singleton instance")
embedder = E5Embedder()