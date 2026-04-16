from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ACTIVE_WORLD_START = "<world>"
ACTIVE_WORLD_END = "</world>"
ACTIVE_SPATIAL = "<spatial>"


def get_active_latent_tokens(latent_token_count: int) -> list[str]:
    return [f"<latent_{idx}>" for idx in range(latent_token_count)]


def get_active_special_tokens(latent_token_count: int) -> list[str]:
    return [
        ACTIVE_WORLD_START,
        ACTIVE_WORLD_END,
        ACTIVE_SPATIAL,
        *get_active_latent_tokens(latent_token_count),
    ]


def build_spatial_sequence(latent_token_count: int) -> str:
    return "".join(
        [ACTIVE_WORLD_START, *get_active_latent_tokens(latent_token_count), ACTIVE_WORLD_END]
    )


def inject_spatial_placeholder(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith(ACTIVE_SPATIAL) or stripped.startswith(ACTIVE_WORLD_START):
        return text
    return f"{ACTIVE_SPATIAL}{text}"


def replace_spatial_placeholder(text: str, latent_token_count: int) -> str:
    if ACTIVE_SPATIAL not in text:
        return text
    return text.replace(ACTIVE_SPATIAL, build_spatial_sequence(latent_token_count), 1)


def add_active_tokens_to_tokenizer(tokenizer, latent_token_count: int) -> int:
    vocab = tokenizer.get_vocab()
    tokens_to_add = [
        token for token in get_active_special_tokens(latent_token_count) if token not in vocab
    ]
    if not tokens_to_add:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})


def configure_activeqwen_config(config, tokenizer, latent_token_count: int) -> None:
    config.activeqwen_enable = True
    config.activeqwen_latent_token_count = latent_token_count
    config.activeqwen_world_start_id = tokenizer.convert_tokens_to_ids(ACTIVE_WORLD_START)
    config.activeqwen_world_end_id = tokenizer.convert_tokens_to_ids(ACTIVE_WORLD_END)
    config.activeqwen_spatial_id = tokenizer.convert_tokens_to_ids(ACTIVE_SPATIAL)
    config.activeqwen_latent_token_ids = [
        tokenizer.convert_tokens_to_ids(token)
        for token in get_active_latent_tokens(latent_token_count)
    ]
    config.architectures = ["ActiveQwen3VLForConditionalGeneration"]


def load_active_embedding(path: str | Path) -> torch.Tensor:
    data = np.load(str(path))
    embedding = data["embedding"]
    target = torch.from_numpy(embedding).to(torch.float32)
    if target.ndim != 4:
        raise ValueError(f"Unexpected latent embedding shape: {target.shape}")
    target = target.squeeze(0)
    if target.ndim != 3:
        raise ValueError(f"Unexpected latent embedding shape after squeeze: {target.shape}")
    return target.contiguous()


def split_by_counts(items: Iterable[torch.Tensor], counts: torch.Tensor) -> list[list[torch.Tensor]]:
    grouped_items: list[list[torch.Tensor]] = []
    item_list = list(items)
    cursor = 0
    for count in counts.tolist():
        grouped_items.append(item_list[cursor : cursor + count])
        cursor += count
    return grouped_items
