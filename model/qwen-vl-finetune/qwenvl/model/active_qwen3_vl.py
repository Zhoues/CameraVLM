from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    GenerationMixin,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import is_torchdynamo_compiling
except ImportError:
    from transformers.utils import is_torchdynamo_compiling

from qwenvl.active import split_by_counts


def _debug_timing_enabled() -> bool:
    return os.environ.get("ACTIVEQWEN_DEBUG_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str) -> None:
    if _debug_timing_enabled():
        rank = os.environ.get("RANK", "?")
        print(f"[activeqwen][rank{rank}] {message}", flush=True)


class ActiveProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_dim: int,
        latent_token_count: int,
        prompt_length: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if latent_token_count <= 0:
            raise ValueError("latent_token_count must be positive")
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")

        self.latent_token_count = latent_token_count
        self.prompt_length = prompt_length

        self.prompt_mlp = nn.Sequential(
            nn.Linear(latent_token_count, prompt_length),
            nn.GELU(),
            nn.Linear(prompt_length, prompt_length),
        )

        projector_layers: list[nn.Module] = [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            projector_layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        projector_layers.append(nn.Linear(hidden_size, target_dim))
        self.hidden_projector = nn.Sequential(*projector_layers)

    def _prompt_mlp_functional(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(
            hidden_states,
            self.prompt_mlp[0].weight,
            self.prompt_mlp[0].bias,
        )
        hidden_states = F.gelu(hidden_states)
        hidden_states = F.linear(
            hidden_states,
            self.prompt_mlp[2].weight,
            self.prompt_mlp[2].bias,
        )
        return hidden_states

    def _hidden_projector_functional(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_projector:
            if isinstance(layer, nn.Linear):
                hidden_states = F.linear(hidden_states, layer.weight, layer.bias)
            elif isinstance(layer, nn.GELU):
                hidden_states = F.gelu(hidden_states, approximate=layer.approximate)
            else:
                raise TypeError(f"Unsupported projector layer type: {type(layer)!r}")
        return hidden_states

    def forward(
        self,
        latent_hidden_states: torch.Tensor,
        image_embeddings: list[torch.Tensor],
        target_seq_len: int,
    ) -> torch.Tensor:
        if latent_hidden_states.shape[0] != self.latent_token_count:
            raise ValueError(
                f"Expected {self.latent_token_count} latent tokens, got {latent_hidden_states.shape[0]}"
            )
        if not image_embeddings:
            raise ValueError("Active projector requires at least one image embedding")

        prompt_tokens = self.prompt_mlp(latent_hidden_states.transpose(0, 1)).transpose(0, 1)
        predictions = []
        for view_embed in image_embeddings:
            joint_tokens = torch.cat([prompt_tokens, view_embed], dim=0)
            pooled_tokens = F.adaptive_avg_pool1d(
                joint_tokens.transpose(0, 1).unsqueeze(0),
                target_seq_len,
            ).squeeze(0).transpose(0, 1)
            predictions.append(self.hidden_projector(pooled_tokens))

        return torch.stack(predictions, dim=0)

    def forward_batch(
        self,
        latent_hidden_states_batch: list[torch.Tensor],
        image_embeddings_batch: list[list[torch.Tensor]],
        target_seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.hidden_projector[-1].weight.device
        dtype = self.hidden_projector[-1].weight.dtype
        hidden_size = self.hidden_projector[0].in_features

        pooled_tokens: list[torch.Tensor] = []
        pair_sample_indices: list[int] = []
        pair_is_real: list[bool] = []

        if latent_hidden_states_batch:
            latent_batch = torch.stack(latent_hidden_states_batch, dim=0).to(device=device, dtype=dtype)
            latent_batch_2d = latent_batch.transpose(1, 2).reshape(-1, self.latent_token_count)
            prompt_tokens_batch = self.prompt_mlp(latent_batch_2d).view(
                latent_batch.shape[0],
                hidden_size,
                self.prompt_length,
            ).transpose(1, 2)

            for sample_idx, (view_embeddings, target_seq_len) in enumerate(
                zip(image_embeddings_batch, target_seq_lens)
            ):
                prompt_tokens = prompt_tokens_batch[sample_idx]
                for view_embed in view_embeddings:
                    joint_tokens = torch.cat(
                        [prompt_tokens, view_embed.to(device=device, dtype=dtype)],
                        dim=0,
                    )
                    pooled = F.adaptive_avg_pool1d(
                        joint_tokens.transpose(0, 1).unsqueeze(0),
                        target_seq_len,
                    ).squeeze(0).transpose(0, 1)
                    pooled_tokens.append(pooled)
                    pair_sample_indices.append(sample_idx)
                    pair_is_real.append(True)

        if not pooled_tokens:
            dummy_latent = torch.zeros(
                (1, self.latent_token_count, hidden_size),
                device=device,
                dtype=dtype,
            )
            dummy_latent_2d = dummy_latent.transpose(1, 2).reshape(-1, self.latent_token_count)
            prompt_tokens = self.prompt_mlp(dummy_latent_2d).view(
                1,
                hidden_size,
                self.prompt_length,
            ).transpose(1, 2)[0]
            dummy_view = torch.zeros((1, hidden_size), device=device, dtype=dtype)
            dummy_joint = torch.cat([prompt_tokens, dummy_view], dim=0)
            dummy_pooled = F.adaptive_avg_pool1d(
                dummy_joint.transpose(0, 1).unsqueeze(0),
                1,
            ).squeeze(0).transpose(0, 1)
            pooled_tokens = [dummy_pooled]
            pair_sample_indices = [0]
            pair_is_real = [False]

        max_target_seq_len = max(token.shape[0] for token in pooled_tokens)
        batch_size = len(pooled_tokens)
        pooled_batch = torch.zeros(
            (batch_size, max_target_seq_len, hidden_size),
            device=device,
            dtype=dtype,
        )
        token_mask = torch.zeros(
            (batch_size, max_target_seq_len),
            device=device,
            dtype=torch.bool,
        )
        for pair_idx, pooled in enumerate(pooled_tokens):
            pooled_batch[pair_idx, : pooled.shape[0]] = pooled
            token_mask[pair_idx, : pooled.shape[0]] = True

        flattened_pooled = pooled_batch.reshape(-1, hidden_size)
        predictions = self.hidden_projector(flattened_pooled).view(
            batch_size,
            max_target_seq_len,
            -1,
        )
        return (
            predictions,
            token_mask,
            torch.tensor(pair_sample_indices, device=device, dtype=torch.long),
            torch.tensor(pair_is_real, device=device, dtype=torch.bool),
        )


@dataclass
class ActiveQwen3VLModelOutputWithPast(Qwen3VLModelOutputWithPast):
    image_feature_splits: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class ActiveQwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    rope_deltas: Optional[torch.LongTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    active_3d_loss: Optional[torch.FloatTensor] = None


class ActiveQwen3VLModel(Qwen3VLModel):
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, ActiveQwen3VLModelOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        image_feature_splits: Optional[tuple[torch.FloatTensor, ...]] = None

        if pixel_values is not None:
            image_feature_outputs = self.get_image_features(pixel_values, image_grid_thw)
            if hasattr(image_feature_outputs, "pooler_output"):
                image_embeds = image_feature_outputs.pooler_output
                deepstack_image_embeds = image_feature_outputs.deepstack_features
            else:
                image_embeds, deepstack_image_embeds = image_feature_outputs
            image_feature_splits = tuple(image_embeds)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            deepstack_image_embeds = None

        if pixel_values_videos is not None:
            video_feature_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            if hasattr(video_feature_outputs, "pooler_output"):
                video_embeds = video_feature_outputs.pooler_output
                deepstack_video_embeds = video_feature_outputs.deepstack_features
            else:
                video_embeds, deepstack_video_embeds = video_feature_outputs
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        else:
            deepstack_video_embeds = None

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return ActiveQwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            image_feature_splits=image_feature_splits,
        )


class ActiveQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = ActiveQwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.active_projector = ActiveProjector(
            hidden_size=config.text_config.hidden_size,
            target_dim=getattr(config, "activeqwen_target_dim", config.text_config.hidden_size),
            latent_token_count=getattr(config, "activeqwen_latent_token_count", 12),
            prompt_length=getattr(config, "activeqwen_projector_prompt_length", 64),
            depth=getattr(config, "activeqwen_projector_depth", 2),
        )
        self.post_init()

    def reset_active_projector(self) -> None:
        self.active_projector = ActiveProjector(
            hidden_size=self.config.text_config.hidden_size,
            target_dim=getattr(self.config, "activeqwen_target_dim", self.config.text_config.hidden_size),
            latent_token_count=getattr(self.config, "activeqwen_latent_token_count", 12),
            prompt_length=getattr(self.config, "activeqwen_projector_prompt_length", 64),
            depth=getattr(self.config, "activeqwen_projector_depth", 2),
        ).to(device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def _compute_zero_active_aux_loss(self) -> torch.Tensor:
        predictions, _, _, _ = self.active_projector.forward_batch([], [], [])
        return predictions.sum() * 0.0

    def _compute_active_3d_loss(
        self,
        hidden_states: torch.Tensor,
        image_feature_splits: Optional[tuple[torch.FloatTensor, ...]],
        active_token_mask: Optional[torch.Tensor],
        active_sample_has_label: Optional[torch.Tensor],
        active_sample_image_counts: Optional[torch.Tensor],
        active_sample_latent_counts: Optional[torch.Tensor],
        active_target_embeddings: Optional[torch.Tensor],
        active_target_view_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        start_time = time.perf_counter()
        pair_latent_hidden: list[torch.Tensor] = []
        pair_views: list[list[torch.Tensor]] = []
        pair_target_seq_lens: list[int] = []
        pair_targets: list[torch.Tensor] = []
        pair_bucket_indices: list[int] = []
        sample_bucket_count = 0

        if (
            image_feature_splits is not None
            and active_token_mask is not None
            and active_sample_has_label is not None
            and active_sample_image_counts is not None
            and active_sample_latent_counts is not None
            and active_target_embeddings is not None
            and active_target_view_mask is not None
        ):
            latent_hidden_states = hidden_states[active_token_mask.to(hidden_states.device)]
            positive_latent_counts = active_sample_latent_counts[active_sample_has_label].tolist()
            if latent_hidden_states.shape[0] != sum(positive_latent_counts):
                raise ValueError(
                    "Active latent token count does not match the hidden-state mask. "
                    f"mask={latent_hidden_states.shape[0]}, expected={sum(positive_latent_counts)}"
                )

            latent_hidden_splits = list(torch.split(latent_hidden_states, positive_latent_counts, dim=0))
            image_feature_groups = split_by_counts(image_feature_splits, active_sample_image_counts.cpu())

            active_idx = 0
            for sample_idx, has_label in enumerate(active_sample_has_label.tolist()):
                if not has_label:
                    continue

                sample_latent_hidden = latent_hidden_splits[active_idx]
                active_idx += 1
                if sample_latent_hidden.shape[0] != self.config.activeqwen_latent_token_count:
                    continue

                sample_targets = active_target_embeddings[active_idx - 1][active_target_view_mask[active_idx - 1]]
                sample_views = image_feature_groups[sample_idx]
                view_count = min(len(sample_views), sample_targets.shape[0])
                if view_count <= 0:
                    continue

                for view_idx in range(view_count):
                    pair_latent_hidden.append(sample_latent_hidden)
                    # NOTE(zhouenshen): 把 view 侧 embedding 做了 detach()；也就是 3D loss 现在还会回传到 LLM/latent 侧和 active_projector，但不会再回传到 visual 分支.
                    pair_views.append(
                        [sample_views[view_idx].detach().to(hidden_states.device, hidden_states.dtype)]
                    )
                    pair_target_seq_lens.append(int(sample_targets[view_idx].shape[0]))
                    pair_targets.append(sample_targets[view_idx])
                    pair_bucket_indices.append(sample_bucket_count)
                sample_bucket_count += 1

        local_pair_count = len(pair_latent_hidden)
        max_pair_count = local_pair_count
        if dist.is_available() and dist.is_initialized():
            pair_count_tensor = torch.tensor([local_pair_count], device=hidden_states.device, dtype=torch.long)
            dist.all_reduce(pair_count_tensor, op=dist.ReduceOp.MAX)
            max_pair_count = int(pair_count_tensor.item())

        _debug_log(
            "_compute_active_3d_loss prepared "
            f"local_pairs={local_pair_count} max_pairs={max_pair_count} "
            f"valid_samples={sample_bucket_count} elapsed={time.perf_counter() - start_time:.2f}s"
        )

        if max_pair_count == 0:
            _debug_log("_compute_active_3d_loss using zero aux path because all ranks have zero active pairs")
            return self._compute_zero_active_aux_loss()

        hidden_size = hidden_states.shape[-1]
        dummy_latent = torch.zeros(
            (self.config.activeqwen_latent_token_count, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        dummy_view = torch.zeros((1, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
        while len(pair_latent_hidden) < max_pair_count:
            pair_latent_hidden.append(dummy_latent)
            pair_views.append([dummy_view])
            pair_target_seq_lens.append(1)
            pair_targets.append(torch.zeros((1, self.config.activeqwen_target_dim), device=hidden_states.device, dtype=hidden_states.dtype))
            pair_bucket_indices.append(-1)

        predictions, token_mask, _, _ = self.active_projector.forward_batch(
            pair_latent_hidden,
            pair_views,
            pair_target_seq_lens,
        )

        _debug_log(
            "_compute_active_3d_loss projector_done "
            f"pair_count={predictions.shape[0]} token_count={predictions.shape[1]} "
            f"elapsed={time.perf_counter() - start_time:.2f}s"
        )

        target_batch = torch.zeros_like(predictions)
        target_token_mask = torch.zeros_like(token_mask)
        real_pair_mask = torch.zeros(predictions.shape[0], device=predictions.device, dtype=torch.bool)
        for pair_idx in range(local_pair_count):
            target = pair_targets[pair_idx].to(predictions.device, predictions.dtype)
            target_batch[pair_idx, : target.shape[0]] = target
            target_token_mask[pair_idx, : target.shape[0]] = True
            real_pair_mask[pair_idx] = True

        token_loss = (predictions - target_batch).pow(2).mean(dim=-1)
        pair_loss = (
            (token_loss * target_token_mask.to(token_loss.dtype)).sum(dim=1)
            / target_token_mask.sum(dim=1).clamp_min(1).to(token_loss.dtype)
        )

        losses = []
        for sample_idx in range(sample_bucket_count):
            sample_pair_mask = real_pair_mask & (
                torch.tensor(pair_bucket_indices, device=predictions.device, dtype=torch.long) == sample_idx
            )
            if torch.any(sample_pair_mask):
                losses.append(pair_loss[sample_pair_mask].mean())

        if not losses:
            _debug_log("_compute_active_3d_loss using zero aux path because no valid sample losses were produced")
            return predictions.sum() * 0.0

        _debug_log(
            "_compute_active_3d_loss complete "
            f"elapsed={time.perf_counter() - start_time:.2f}s"
        )
        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        active_token_mask: Optional[torch.Tensor] = None,
        active_sample_has_label: Optional[torch.Tensor] = None,
        active_sample_image_counts: Optional[torch.Tensor] = None,
        active_sample_latent_counts: Optional[torch.Tensor] = None,
        active_target_embeddings: Optional[torch.Tensor] = None,
        active_target_view_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[tuple, ActiveQwen3VLCausalLMOutputWithPast]:
        start_time = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        _debug_log(f"forward model_done elapsed={time.perf_counter() - start_time:.2f}s")

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        _debug_log(f"forward lm_head_done elapsed={time.perf_counter() - start_time:.2f}s")

        text_loss = None
        if labels is not None:
            text_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        _debug_log(f"forward text_loss_done elapsed={time.perf_counter() - start_time:.2f}s")

        active_3d_loss = self._compute_active_3d_loss(
            hidden_states=hidden_states,
            image_feature_splits=outputs.image_feature_splits,
            active_token_mask=active_token_mask,
            active_sample_has_label=active_sample_has_label,
            active_sample_image_counts=active_sample_image_counts,
            active_sample_latent_counts=active_sample_latent_counts,
            active_target_embeddings=active_target_embeddings,
            active_target_view_mask=active_target_view_mask,
        )
        _debug_log(f"forward active_3d_loss_done elapsed={time.perf_counter() - start_time:.2f}s")

        loss = None
        if text_loss is not None:
            loss = getattr(self.config, "active_ce_loss_weight", 1.0) * text_loss
        if active_3d_loss is not None:
            weighted_active_loss = getattr(self.config, "active_3d_loss_weight", 1.0) * active_3d_loss
            loss = weighted_active_loss if loss is None else loss + weighted_active_loss

        _debug_log(f"forward return elapsed={time.perf_counter() - start_time:.2f}s")

        return ActiveQwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
            text_loss=text_loss,
            active_3d_loss=active_3d_loss,
        )
