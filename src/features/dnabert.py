"""Off-the-shelf Hugging Face DNA embeddings (no fine-tuning).

Optional encoder: ``encoder.name: dnabert``. ``fit()`` is a no-op -- the
point is to drop in a frozen DNABERT-style model and keep the same
fallback. Requires ``transformers`` + ``torch`` (not in the default
``requirements.txt``; install only if you want this path).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def dnabert_params(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("dnabert") if isinstance(config.get("dnabert"), dict) else {}
    merged = {**config, **nested}
    return {
        "model_id": str(merged.get("model_id", "zhihan1996/DNABERT-2-117M")),
        "max_len": int(merged.get("max_len", 256)),
        "batch_size": int(merged.get("batch_size", 8)),
        "device": str(merged.get("device", "cpu")),
        "trust_remote_code": bool(merged.get("trust_remote_code", True)),
    }


class HuggingFaceDNAEncoder:
    """Mean-pool a frozen AutoModel over nucleotide tokens."""

    def __init__(
        self,
        model_id: str = "zhihan1996/DNABERT-2-117M",
        max_len: int = 256,
        batch_size: int = 8,
        device: str = "cpu",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._model = None
        self._fitted = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "encoder.name=dnabert requires the optional packages torch and "
                "transformers. Install them, or use encoder.name=conv_triplet "
                "(numpy CNN, no extra deps)."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def fit(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> "HuggingFaceDNAEncoder":
        # Frozen off-the-shelf embeddings: load weights, do not update them.
        del sequences, labels
        self._load()
        self._fitted = True
        return self

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        if not self._fitted or self._model is None or self._tokenizer is None:
            raise RuntimeError("HuggingFaceDNAEncoder.fit() must be called before transform()")
        seqs = [s.upper() for s in sequences]
        if not seqs:
            hidden = int(self._model.config.hidden_size)
            return np.zeros((0, hidden), dtype=np.float64)

        torch = self._torch
        chunks: list[np.ndarray] = []
        for start in range(0, len(seqs), self.batch_size):
            batch = seqs[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self._model(**encoded)
                hidden = out.last_hidden_state
                mask = encoded.get("attention_mask")
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    weights = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(chunks, axis=0)

    def fit_transform(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> np.ndarray:
        seqs = list(sequences)
        return self.fit(seqs, labels).transform(seqs)
