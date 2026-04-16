"""
source /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate anno

CUDA_VISIBLE_DEVICES=0 bash /share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/API/start_activeqwen3_openai_batched.sh \
    --checkpoint /share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/model/qwen-vl-finetune/output/activeqwen3vl_2b_2_nodes_gas_4_camera+visual_search+general/checkpoint-12000 \
    --processor-path /share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/model/qwen-vl-finetune/output/activeqwen3vl_2b_ap_pap_hstar \
    --served-model-name activeqwen3vl \
    --host 127.0.0.1 \
    --port 25546 \
    --max-new-tokens-default 2048 \
    --max-batch-size 8 \
    --batch-wait-ms 20
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoProcessor

from qwenvl.model import ActiveQwen3VLForConditionalGeneration


logger = logging.getLogger(__name__)


class ImageURLPayload(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURLPayload | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentItem]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    temperature: float | None = 0.0
    stream: bool = False


@dataclass
class PendingRequest:
    hf_messages: list[dict[str, Any]]
    max_new_tokens: int
    temperature: float
    do_sample: bool
    created_at: float = field(default_factory=time.time)
    done_event: threading.Event = field(default_factory=threading.Event)
    output_text: str | None = None
    usage: dict[str, int] | None = None
    error_message: str | None = None

    @property
    def batch_key(self) -> tuple[int, float, bool]:
        return (self.max_new_tokens, round(self.temperature, 6), self.do_sample)


class ServerState:
    def __init__(
        self,
        checkpoint_path: str,
        processor_path: str | None,
        served_model_name: str,
        device_map: str,
        dtype: str,
        attn_implementation: str | None,
        max_new_tokens_default: int,
        max_batch_size: int,
        batch_wait_ms: int,
        request_timeout_sec: float,
        max_queue_size: int,
    ):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.processor_path = self._resolve_processor_path(processor_path)
        self.served_model_name = served_model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens_default = int(max_new_tokens_default)
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_ms = int(batch_wait_ms)
        self.request_timeout_sec = float(request_timeout_sec)
        queue_size = 0 if int(max_queue_size) <= 0 else int(max_queue_size)
        self.request_queue: queue.Queue[PendingRequest] = queue.Queue(maxsize=queue_size)
        self.deferred_queue: deque[PendingRequest] = deque()
        self.shutdown_event = threading.Event()
        self.total_batches = 0
        self.total_requests = 0

        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model = self._load_model()
        self.model.eval()

        self.batch_worker = threading.Thread(target=self._batch_loop, daemon=True, name="activeqwen3vl-batch-worker")
        self.batch_worker.start()

    def _resolve_processor_path(self, processor_path: str | None) -> str:
        if processor_path:
            return str(Path(processor_path).expanduser())
        checkpoint_path = Path(self.checkpoint_path)
        if (checkpoint_path / "preprocessor_config.json").is_file():
            return str(checkpoint_path)
        raise ValueError(
            "processor_path is required because the checkpoint directory does not contain preprocessor_config.json"
        )

    def _load_model(self):
        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if self.dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "float32":
            kwargs["torch_dtype"] = torch.float32

        if self.attn_implementation and self.attn_implementation != "none":
            kwargs["attn_implementation"] = self.attn_implementation

        try:
            return ActiveQwen3VLForConditionalGeneration.from_pretrained(self.checkpoint_path, **kwargs)
        except Exception:
            if "attn_implementation" not in kwargs:
                raise
            kwargs.pop("attn_implementation")
            return ActiveQwen3VLForConditionalGeneration.from_pretrained(self.checkpoint_path, **kwargs)

    def shutdown(self) -> None:
        self.shutdown_event.set()
        if self.batch_worker.is_alive():
            self.batch_worker.join(timeout=5)

    def infer(
        self,
        hf_messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        pending = PendingRequest(
            hf_messages=hf_messages,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=float(temperature) > 1e-6,
        )
        try:
            self.request_queue.put(pending, timeout=1.0)
        except queue.Full as exc:
            raise RuntimeError("request queue is full, please retry later") from exc

        if not pending.done_event.wait(timeout=self.request_timeout_sec):
            raise TimeoutError(
                f"request timed out after waiting {self.request_timeout_sec:.1f}s for batched inference"
            )
        if pending.error_message:
            raise RuntimeError(pending.error_message)
        if pending.output_text is None or pending.usage is None:
            raise RuntimeError("batched inference finished without a result payload")
        return pending.output_text, pending.usage

    def _take_next_request(self, timeout: float | None) -> PendingRequest | None:
        if self.deferred_queue:
            return self.deferred_queue.popleft()
        if self.shutdown_event.is_set():
            return None
        try:
            if timeout is None:
                return self.request_queue.get(timeout=0.1)
            return self.request_queue.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None

    def _collect_batch(self) -> list[PendingRequest]:
        first = self._take_next_request(timeout=None)
        if first is None:
            return []

        batch = [first]
        deadline = time.time() + (self.batch_wait_ms / 1000.0)
        batch_key = first.batch_key

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            candidate = self._take_next_request(timeout=remaining)
            if candidate is None:
                break
            if candidate.batch_key == batch_key:
                batch.append(candidate)
            else:
                self.deferred_queue.append(candidate)
                if self.deferred_queue and self.deferred_queue[0].batch_key == batch_key:
                    continue
                break
        return batch

    def _prepare_inputs(self, batched_messages: list[list[dict[str, Any]]]) -> dict[str, Any]:
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        }
        device = self.model.device
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if key not in allowed_keys:
                continue
            prepared[key] = value.to(device) if hasattr(value, "to") else value
        return prepared

    def _decode_generation(
        self,
        inputs: dict[str, Any],
        generated_ids: torch.Tensor,
    ) -> list[tuple[str, dict[str, int]]]:
        if "attention_mask" in inputs:
            prompt_lengths = [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
        else:
            prompt_lengths = [int(row.shape[-1]) for row in inputs["input_ids"]]

        completion_ids = [
            generated_ids[row_idx, prompt_len:]
            for row_idx, prompt_len in enumerate(prompt_lengths)
        ]
        output_texts = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        decoded: list[tuple[str, dict[str, int]]] = []
        for output_text, prompt_len, completion in zip(output_texts, prompt_lengths, completion_ids):
            completion_tokens = int(completion.shape[-1])
            decoded.append(
                (
                    output_text,
                    {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_len + completion_tokens,
                    },
                )
            )
        return decoded

    def _process_batch(self, batch: list[PendingRequest]) -> None:
        batch_started_at = time.time()
        batched_messages = [item.hf_messages for item in batch]
        seed = batch[0]
        inputs = self._prepare_inputs(batched_messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=seed.max_new_tokens,
                temperature=(seed.temperature if seed.do_sample else 1.0),
                do_sample=seed.do_sample,
            )
        decoded = self._decode_generation(inputs, generated_ids)
        self.total_batches += 1
        self.total_requests += len(batch)
        logger.info(
            "processed batch_size=%d max_new_tokens=%d do_sample=%s wait_ms=%d latency_sec=%.3f",
            len(batch),
            seed.max_new_tokens,
            seed.do_sample,
            self.batch_wait_ms,
            time.time() - batch_started_at,
        )
        for pending, (text, usage) in zip(batch, decoded):
            pending.output_text = text
            pending.usage = usage
            pending.done_event.set()

    def _fail_batch(self, batch: list[PendingRequest], exc: Exception) -> None:
        error_message = f"inference failed: {exc}\n{traceback.format_exc()}"
        for pending in batch:
            pending.error_message = error_message
            pending.done_event.set()

    def _batch_loop(self) -> None:
        while not self.shutdown_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._process_batch(batch)
            except Exception as exc:
                logger.exception("batched inference failed")
                self._fail_batch(batch, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ActiveQwen3VL OpenAI-compatible batched server")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--processor-path", type=str, default=None, help="processor/tokenizer path")
    parser.add_argument("--served-model-name", type=str, default="activeqwen3vl")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=25556)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max-new-tokens-default", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=int, default=20)
    parser.add_argument("--request-timeout-sec", type=float, default=600.0)
    parser.add_argument("--max-queue-size", type=int, default=0, help="0 means unbounded")
    return parser.parse_args()


def _decode_image_url(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        _, encoded = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    path = Path(url).expanduser()
    if path.is_file():
        return Image.open(path).convert("RGB")

    raise ValueError(f"Unsupported image URL: {url}")


def _convert_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message.content, str):
            converted.append(
                {
                    "role": message.role,
                    "content": [{"type": "text", "text": message.content}],
                }
            )
            continue

        items: list[dict[str, Any]] = []
        for item in message.content:
            if item.type == "text":
                items.append({"type": "text", "text": item.text or ""})
                continue
            if item.image_url is None:
                raise ValueError("image_url content item is missing image_url payload")
            items.append({"type": "image", "image": _decode_image_url(item.image_url.url)})
        converted.append({"role": message.role, "content": items})
    return converted


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="ActiveQwen3VL OpenAI Batched Server")

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        state.shutdown()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": state.served_model_name,
            "checkpoint": state.checkpoint_path,
            "processor_path": state.processor_path,
            "max_batch_size": state.max_batch_size,
            "batch_wait_ms": state.batch_wait_ms,
            "request_timeout_sec": state.request_timeout_sec,
            "queue_depth": state.request_queue.qsize() + len(state.deferred_queue),
            "total_batches": state.total_batches,
            "total_requests": state.total_requests,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "activeqwen3vl",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if request.model and request.model != state.served_model_name:
            raise HTTPException(
                status_code=400,
                detail=f"requested model {request.model} does not match served model {state.served_model_name}",
            )

        try:
            hf_messages = _convert_messages(request.messages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        max_new_tokens = int(request.max_tokens or state.max_new_tokens_default)
        temperature = float(request.temperature or 0.0)

        try:
            output_text, usage = state.infer(
                hf_messages=hf_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": state.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = parse_args()
    state = ServerState(
        checkpoint_path=args.checkpoint,
        processor_path=args.processor_path,
        served_model_name=args.served_model_name,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens_default=args.max_new_tokens_default,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        request_timeout_sec=args.request_timeout_sec,
        max_queue_size=args.max_queue_size,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
