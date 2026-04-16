"""
source /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate anno

bash /share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/API/start_activeqwen3_openai.sh \
    --checkpoint /path/to/your/checkpoint \
    --processor-path /path/to/your/processor \
    --served-model-name activeqwen3vl \
    --host 0.0.0.0 \
    --port 25546 \
    --max-new-tokens-default 2048
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time
import traceback
import uuid
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
    ):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.processor_path = self._resolve_processor_path(processor_path)
        self.served_model_name = served_model_name
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens_default = int(max_new_tokens_default)
        self.lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model = self._load_model()
        self.model.eval()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ActiveQwen3VL OpenAI-compatible server")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--processor-path", type=str, default=None, help="processor/tokenizer path")
    parser.add_argument("--served-model-name", type=str, default="activeqwen3vl")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=25546)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max-new-tokens-default", type=int, default=2048)
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


def _prepare_inputs(state: ServerState, messages: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = state.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    allowed_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    }
    device = state.model.device
    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        if key not in allowed_keys:
            continue
        prepared[key] = value.to(device) if hasattr(value, "to") else value
    return prepared


def _decode_generation(
    state: ServerState,
    inputs: dict[str, Any],
    generated_ids: torch.Tensor,
) -> tuple[str, dict[str, int]]:
    prompt_len = int(inputs["input_ids"].shape[-1])
    completion_ids = generated_ids[:, prompt_len:]
    text = state.processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": int(completion_ids.shape[-1]),
        "total_tokens": int(prompt_len + completion_ids.shape[-1]),
    }
    return text, usage


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="ActiveQwen3VL OpenAI Server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": state.served_model_name,
            "checkpoint": state.checkpoint_path,
            "processor_path": state.processor_path,
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
        do_sample = temperature > 1e-6

        with state.lock:
            try:
                inputs = _prepare_inputs(state, hf_messages)
                with torch.inference_mode():
                    generated_ids = state.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=(temperature if do_sample else 1.0),
                        do_sample=do_sample,
                    )
                output_text, usage = _decode_generation(state, inputs, generated_ids)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"inference failed: {exc}\n{traceback.format_exc()}",
                ) from exc

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
    args = parse_args()
    state = ServerState(
        checkpoint_path=args.checkpoint,
        processor_path=args.processor_path,
        served_model_name=args.served_model_name,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens_default=args.max_new_tokens_default,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
