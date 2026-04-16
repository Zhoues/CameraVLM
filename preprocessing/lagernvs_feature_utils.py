from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent


def _resolve_lagernvs_root() -> Path:
    candidates = [
        THIS_DIR.parent / "lagernvs",  # new location: CameraVLM/lagernvs
        THIS_DIR / "lagernvs",  # backward-compatible fallback
    ]
    for candidate in candidates:
        if (candidate / "models" / "encoder_decoder.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate lagernvs source directory. Tried:\n"
        + "\n".join(str(path) for path in candidates)
    )


LAGERNVS_ROOT = _resolve_lagernvs_root()
DEFAULT_CAMERA_DATA_ROOT = Path(
    "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data"
)
DEFAULT_LATENT_ROOT = Path(
    "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/share/project/zhouenshen/hpfs/ckpt/diffusion/lagernvs_general_512/model.pt"
)

if str(LAGERNVS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAGERNVS_ROOT))

from models.encoder_decoder import EncDec_VitB8  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


SAFE_NAME_PATTERN = re.compile(r"[^\w\-.]+")


@dataclass(frozen=True)
class DatasetPaths:
    dataset_name: str
    metadata_path: str
    camera_image_root: str
    latent_dir: str
    manifest_path: str
    config_path: str
    summary_path: str
    error_dir: str
    progress_dir: str


@dataclass(frozen=True)
class QALatentPaths:
    dataset_name: str
    annotation_path: str
    image_root: str
    latent_dir: str
    sequence_manifest_dir: str
    sequence_manifest_path: str
    qa_output_path: str
    config_path: str
    summary_path: str
    error_dir: str
    progress_dir: str


@dataclass(frozen=True)
class SampleRecord:
    metadata_index: int
    sample_key: str
    image_rel_paths: tuple[str, ...]
    image_abs_paths: tuple[str, ...]
    poses: tuple[tuple[float, float], ...]
    camera: dict[str, Any]
    output_path: str


@dataclass(frozen=True)
class ScanResult:
    metadata_index: int
    status: str
    reason: str


def log_info(message: str) -> None:
    print(f"[lagernvs] {message}", flush=True)


def configure_stdout_for_tqdm() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _preferred_tqdm_stream():
    stderr_is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if stderr_is_tty:
        return sys.stderr
    if stdout_is_tty:
        return sys.stdout
    return sys.stderr


def tqdm_options(disable: bool = False) -> dict[str, Any]:
    return {
        "disable": disable,
        "file": _preferred_tqdm_stream(),
        "dynamic_ncols": True,
        "mininterval": 0.2,
        "leave": True,
        "ascii": True,
    }


def format_count_summary(summary: dict[str, Any]) -> str:
    ordered_keys = [
        "total_records",
        "selected_records",
        "pending_records",
        "processed",
        "failed",
        "skipped_existing",
        "regenerated_invalid_existing",
        "pending_invalid_existing",
        "pending_missing",
        "pending_overwrite_existing_valid",
        "pending_overwrite_existing_invalid",
        "scan_workers",
    ]
    parts: list[str] = []
    for key in ordered_keys:
        if key in summary:
            parts.append(f"{key}={summary[key]}")
    return ", ".join(parts)


def sanitize_filename(text: str) -> str:
    text = str(text or "").strip().replace(os.sep, "_")
    text = re.sub(r"\s+", "_", text)
    text = SAFE_NAME_PATTERN.sub("_", text)
    text = text.strip("._")
    return text or "sample"


def resolve_dataset_paths(
    dataset_name: str | None,
    camera_data_root: str | os.PathLike[str] = DEFAULT_CAMERA_DATA_ROOT,
    latent_root: str | os.PathLike[str] = DEFAULT_LATENT_ROOT,
    metadata_path: str | os.PathLike[str] | None = None,
    camera_image_root: str | os.PathLike[str] | None = None,
) -> DatasetPaths:
    if metadata_path is None:
        if not dataset_name:
            raise ValueError("dataset_name and metadata_path cannot both be empty")
        metadata_path = Path(camera_data_root) / dataset_name / "metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if dataset_name is None:
        dataset_name = metadata_path.parent.name

    if camera_image_root is None:
        camera_image_root = metadata_path.parent / "camera_image"
    else:
        camera_image_root = Path(camera_image_root)

    latent_dir = Path(latent_root) / dataset_name
    error_dir = latent_dir / "_errors"
    return DatasetPaths(
        dataset_name=dataset_name,
        metadata_path=str(metadata_path),
        camera_image_root=str(camera_image_root),
        latent_dir=str(latent_dir),
        manifest_path=str(latent_dir / "manifest.jsonl"),
        config_path=str(latent_dir / "extract_config.json"),
        summary_path=str(latent_dir / "extract_summary.json"),
        error_dir=str(error_dir),
        progress_dir=str(latent_dir / "_progress"),
    )


def ensure_output_dirs(paths: DatasetPaths) -> None:
    Path(paths.latent_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.error_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.progress_dir).mkdir(parents=True, exist_ok=True)


def _load_qwenvl_data_dict() -> dict[str, Any]:
    qwenvl_root = THIS_DIR.parent / "model" / "qwen-vl-finetune"
    if str(qwenvl_root) not in sys.path:
        sys.path.insert(0, str(qwenvl_root))
    module = importlib.import_module("qwenvl.data")
    data_dict = getattr(module, "data_dict", None)
    if not isinstance(data_dict, dict):
        raise RuntimeError("failed to load qwenvl.data.data_dict")
    return data_dict


def resolve_qa_latent_paths(
    dataset_name: str,
    latent_root: str | os.PathLike[str] = DEFAULT_LATENT_ROOT,
    annotation_path: str | os.PathLike[str] | None = None,
    image_root: str | os.PathLike[str] | None = None,
    qa_output_path: str | os.PathLike[str] | None = None,
) -> QALatentPaths:
    dataset_key = str(dataset_name or "").strip().lower()
    if not dataset_key:
        raise ValueError("dataset_name must not be empty")

    dataset_name = str(dataset_name).strip()
    data_dict = _load_qwenvl_data_dict()
    if dataset_key not in data_dict:
        raise KeyError(f"dataset {dataset_name} not found in qwenvl.data.data_dict")
    dataset_cfg = data_dict[dataset_key]

    if annotation_path is None:
        annotation_path = dataset_cfg["annotation_path"]
    if image_root is None:
        image_root = dataset_cfg["data_path"]
    annotation_path = str(Path(annotation_path))
    image_root = str(Path(image_root))

    if qa_output_path is None:
        qa_output_path = str(Path(annotation_path).with_name("QA_v5_w_latent.json"))
    else:
        qa_output_path = str(Path(qa_output_path))

    latent_dir = Path(latent_root) / dataset_name
    sequence_manifest_dir = latent_dir / "_qa_sequence_shards"
    return QALatentPaths(
        dataset_name=dataset_name,
        annotation_path=annotation_path,
        image_root=image_root,
        latent_dir=str(latent_dir),
        sequence_manifest_dir=str(sequence_manifest_dir),
        sequence_manifest_path=str(latent_dir / "qa_sequence_manifest.jsonl"),
        qa_output_path=qa_output_path,
        config_path=str(latent_dir / "qa_extract_config.json"),
        summary_path=str(latent_dir / "qa_extract_summary.json"),
        error_dir=str(latent_dir / "_errors"),
        progress_dir=str(latent_dir / "_progress"),
    )


def load_metadata(metadata_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        raise ValueError(f"metadata must be a list, got {type(metadata)}")
    return metadata


def iter_json_array(
    json_path: str | os.PathLike[str],
    show_progress: bool = True,
    progress_desc: str = "Streaming JSON",
    chunk_size: int = 1 << 20,
):
    json_path = str(json_path)
    total_bytes = os.path.getsize(json_path)
    decoder = json.JSONDecoder()
    with open(json_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            total=total_bytes,
            desc=progress_desc,
            unit="B",
            unit_scale=True,
            **tqdm_options(disable=not show_progress),
        )
        if show_progress:
            pbar.refresh()
        buffer = ""
        start_idx = 0
        array_started = False
        array_finished = False
        yielded = 0

        def _fill_buffer() -> bool:
            nonlocal buffer, start_idx
            chunk = f.read(chunk_size)
            if not chunk:
                return False
            buffer = buffer[start_idx:] + chunk
            start_idx = 0
            pbar.n = min(total_bytes, f.buffer.tell())
            pbar.set_postfix(items=yielded, refresh=True)
            return True

        try:
            while not array_finished:
                if start_idx >= len(buffer):
                    if not _fill_buffer():
                        break

                while start_idx < len(buffer) and buffer[start_idx].isspace():
                    start_idx += 1

                if not array_started:
                    if start_idx >= len(buffer):
                        continue
                    if buffer[start_idx] != "[":
                        raise ValueError(f"{json_path} is not a JSON array")
                    array_started = True
                    start_idx += 1

                while True:
                    while start_idx < len(buffer) and buffer[start_idx].isspace():
                        start_idx += 1
                    if start_idx < len(buffer) and buffer[start_idx] == ",":
                        start_idx += 1
                        continue
                    break

                if start_idx >= len(buffer):
                    if not _fill_buffer():
                        break
                    continue

                if buffer[start_idx] == "]":
                    array_finished = True
                    start_idx += 1
                    break

                try:
                    item, item_end = decoder.raw_decode(buffer, start_idx)
                except json.JSONDecodeError:
                    if not _fill_buffer():
                        raise
                    continue

                yielded += 1
                start_idx = item_end
                yield item
        finally:
            final_bytes = total_bytes if array_finished else min(total_bytes, f.buffer.tell())
            pbar.n = final_bytes
            pbar.set_postfix(items=yielded, refresh=True)
            pbar.close()


class JsonArrayWriter:
    def __init__(self, output_path: str | os.PathLike[str]):
        self.output_path = Path(output_path)
        self.tmp_path = self.output_path.with_name(self.output_path.name + ".tmp")
        self.file = None
        self.is_first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.tmp_path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.is_first = True
        return self

    def write(self, item: Any) -> None:
        if self.file is None:
            raise RuntimeError("JsonArrayWriter is not opened")
        if not self.is_first:
            self.file.write(",\n")
        json.dump(item, self.file, ensure_ascii=False)
        self.is_first = False

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.file is not None:
            self.file.write("\n]\n")
            self.file.close()
            self.file = None
        if exc_type is None:
            os.replace(self.tmp_path, self.output_path)
        else:
            try:
                self.tmp_path.unlink()
            except OSError:
                pass


def hash_image_sequence(image_refs: Sequence[str]) -> str:
    canonical = "\n".join(str(ref) for ref in image_refs)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def slice_metadata(
    metadata: Sequence[dict[str, Any]],
    start_idx: int = 0,
    end_idx: int | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    total = len(metadata)
    start = max(0, int(start_idx))
    end = total if end_idx is None else min(total, int(end_idx))
    sliced = list(metadata[start:end])
    if max_samples is not None:
        sliced = sliced[: max(0, int(max_samples))]
    return sliced, start, end


def _derive_sample_key(item: dict[str, Any], metadata_index: int) -> str:
    folder_name = str(item.get("folder_name", "")).strip()
    if folder_name:
        return sanitize_filename(folder_name)

    image_refs = item.get("images", [])
    if isinstance(image_refs, list) and image_refs:
        first_parent = Path(str(image_refs[0])).parent.name
        if first_parent and first_parent != ".":
            return sanitize_filename(first_parent)

    prefix_name = str(item.get("prefix_folder_name", "")).strip()
    if prefix_name:
        return sanitize_filename(prefix_name)

    task_id = item.get("task_id")
    sample_id = item.get("sample_id")
    if task_id is not None and sample_id is not None:
        return sanitize_filename(
            f"{item.get('scene_id', 'scene')}_{item.get('object', 'obj')}_{task_id}_{sample_id}"
        )

    return f"sample_{metadata_index:06d}"


def _extract_pose_list(item: dict[str, Any], num_images: int) -> tuple[tuple[float, float], ...]:
    poses = item.get("poses", [])
    pose_list: list[tuple[float, float]] = []

    if isinstance(poses, list):
        for pose in poses[:num_images]:
            if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                pose_list.append((float(pose[0]), float(pose[1])))

    if len(pose_list) == num_images:
        return tuple(pose_list)

    frames = item.get("frames", [])
    if isinstance(frames, list):
        pose_list = []
        for frame in frames[:num_images]:
            if isinstance(frame, dict):
                pose_list.append(
                    (float(frame.get("yaw", 0.0)), float(frame.get("pitch", 0.0)))
                )
        if len(pose_list) == num_images:
            return tuple(pose_list)

    if pose_list:
        while len(pose_list) < num_images:
            pose_list.append(pose_list[-1])
        return tuple(pose_list[:num_images])

    return tuple((0.0, 0.0) for _ in range(num_images))


def _dedupe_sample_record_keys(
    records: Sequence[SampleRecord],
    latent_dir: str | os.PathLike[str],
) -> list[SampleRecord]:
    latent_dir = str(latent_dir)
    key_counts: dict[str, int] = {}
    deduped_records: list[SampleRecord] = []
    for record in records:
        sample_key = record.sample_key
        key_counts[sample_key] = key_counts.get(sample_key, 0) + 1
        if key_counts[sample_key] > 1:
            sample_key = f"{sample_key}__idx{record.metadata_index:06d}"
        deduped_records.append(
            SampleRecord(
                metadata_index=record.metadata_index,
                sample_key=sample_key,
                image_rel_paths=record.image_rel_paths,
                image_abs_paths=record.image_abs_paths,
                poses=record.poses,
                camera=record.camera,
                output_path=os.path.join(latent_dir, f"{sample_key}.npz"),
            )
        )
    return deduped_records


def _build_sample_records_local(
    metadata: Sequence[dict[str, Any]],
    camera_image_root: str | os.PathLike[str],
    latent_dir: str | os.PathLike[str],
    metadata_offset: int = 0,
    dedupe_keys: bool = True,
) -> list[SampleRecord]:
    camera_image_root = str(camera_image_root)
    latent_dir = str(latent_dir)

    raw_keys = [
        _derive_sample_key(item, metadata_offset + idx)
        for idx, item in enumerate(metadata)
    ]
    records: list[SampleRecord] = []

    for local_index, item in enumerate(metadata):
        metadata_index = metadata_offset + local_index
        image_refs = item.get("images", [])
        if not isinstance(image_refs, list) or not image_refs:
            continue

        image_rel_paths = tuple(str(path) for path in image_refs)
        image_abs_paths = tuple(
            rel_path if os.path.isabs(rel_path) else os.path.join(camera_image_root, rel_path)
            for rel_path in image_rel_paths
        )
        sample_key = raw_keys[local_index]

        camera = item.get("camera", {})
        if not isinstance(camera, dict):
            camera = {}

        records.append(
            SampleRecord(
                metadata_index=metadata_index,
                sample_key=sample_key,
                image_rel_paths=image_rel_paths,
                image_abs_paths=image_abs_paths,
                poses=_extract_pose_list(item, len(image_rel_paths)),
                camera=camera,
                output_path=os.path.join(latent_dir, f"{sample_key}.npz"),
            )
        )
    if dedupe_keys:
        return _dedupe_sample_record_keys(records, latent_dir=latent_dir)
    return records


def build_sample_records(
    metadata: Sequence[dict[str, Any]],
    camera_image_root: str | os.PathLike[str],
    latent_dir: str | os.PathLike[str],
    metadata_offset: int = 0,
) -> list[SampleRecord]:
    return _build_sample_records_local(
        metadata=metadata,
        camera_image_root=camera_image_root,
        latent_dir=latent_dir,
        metadata_offset=metadata_offset,
        dedupe_keys=True,
    )


def _build_sample_records_chunk(
    chunk: Sequence[dict[str, Any]],
    camera_image_root: str,
    latent_dir: str,
    metadata_offset: int,
) -> list[SampleRecord]:
    return _build_sample_records_local(
        metadata=chunk,
        camera_image_root=camera_image_root,
        latent_dir=latent_dir,
        metadata_offset=metadata_offset,
        dedupe_keys=False,
    )


def build_sample_records_multiprocess(
    metadata: Sequence[dict[str, Any]],
    camera_image_root: str | os.PathLike[str],
    latent_dir: str | os.PathLike[str],
    metadata_offset: int = 0,
    num_workers: int = 1,
    show_progress: bool = True,
    progress_desc: str = "Building sample records",
) -> list[SampleRecord]:
    metadata_list = list(metadata)
    total = len(metadata_list)
    if total == 0:
        return []

    camera_image_root = str(camera_image_root)
    latent_dir = str(latent_dir)
    workers = max(1, min(int(num_workers), total))
    if workers == 1:
        return build_sample_records(
            metadata=metadata_list,
            camera_image_root=camera_image_root,
            latent_dir=latent_dir,
            metadata_offset=metadata_offset,
        )

    chunk_size = max(1, total // (workers * 8))
    chunks_with_offsets: list[tuple[list[dict[str, Any]], int]] = []
    for chunk_start in range(0, total, chunk_size):
        chunk = metadata_list[chunk_start : chunk_start + chunk_size]
        chunks_with_offsets.append((chunk, metadata_offset + chunk_start))

    ordered_parts: list[list[SampleRecord] | None] = [None] * len(chunks_with_offsets)
    max_pending_futures = max(32, workers * 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending = set()
        future_to_chunk_idx: dict[Any, int] = {}
        submit_bar = tqdm(
            total=len(chunks_with_offsets),
            desc=f"{progress_desc} submit",
            unit="chunk",
            **tqdm_options(disable=not show_progress),
        )
        done_bar = tqdm(
            total=len(chunks_with_offsets),
            desc=progress_desc,
            unit="chunk",
            **tqdm_options(disable=not show_progress),
        )
        if show_progress:
            submit_bar.refresh()
            done_bar.refresh()
        try:
            for chunk_idx, (chunk, chunk_offset) in enumerate(chunks_with_offsets):
                if len(pending) >= max_pending_futures:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        finished_idx = future_to_chunk_idx.pop(future)
                        ordered_parts[finished_idx] = future.result()
                        done_bar.update(1)

                future = ex.submit(
                    _build_sample_records_chunk,
                    chunk,
                    camera_image_root,
                    latent_dir,
                    chunk_offset,
                )
                pending.add(future)
                future_to_chunk_idx[future] = chunk_idx
                submit_bar.update(1)
                submit_bar.set_postfix(
                    inflight=len(pending),
                    max_pending=max_pending_futures,
                    refresh=False,
                )

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    finished_idx = future_to_chunk_idx.pop(future)
                    ordered_parts[finished_idx] = future.result()
                    done_bar.update(1)
        finally:
            submit_bar.close()
            done_bar.close()

    merged_records: list[SampleRecord] = []
    for part in ordered_parts:
        if part:
            merged_records.extend(part)
    return _dedupe_sample_record_keys(merged_records, latent_dir=latent_dir)


def _sequence_manifest_shard_path(
    sequence_manifest_dir: str | os.PathLike[str],
    shard_idx: int,
) -> Path:
    return Path(sequence_manifest_dir) / f"worker_{int(shard_idx):02d}.jsonl"


def reset_sequence_manifest_dir(
    sequence_manifest_dir: str | os.PathLike[str],
    num_shards: int,
) -> list[Path]:
    sequence_manifest_dir = Path(sequence_manifest_dir)
    sequence_manifest_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        _sequence_manifest_shard_path(sequence_manifest_dir, shard_idx)
        for shard_idx in range(max(1, int(num_shards)))
    ]
    for shard_path in shard_paths:
        try:
            shard_path.unlink()
        except OSError:
            pass
    return shard_paths


def build_qa_latent_manifest(
    qa_input_path: str | os.PathLike[str],
    qa_output_path: str | os.PathLike[str],
    sequence_manifest_path: str | os.PathLike[str],
    sequence_manifest_dir: str | os.PathLike[str],
    num_sequence_shards: int,
    show_progress: bool = True,
    progress_desc: str = "Scanning QA",
    max_samples: int | None = None,
) -> dict[str, Any]:
    qa_input_path = str(qa_input_path)
    qa_output_path = str(qa_output_path)
    sequence_manifest_path = Path(sequence_manifest_path)
    shard_paths = reset_sequence_manifest_dir(sequence_manifest_dir, num_sequence_shards)
    sequence_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    digest_to_name: dict[str, str] = {}
    shard_files = [open(path, "w", encoding="utf-8") for path in shard_paths]
    main_manifest_file = open(sequence_manifest_path, "w", encoding="utf-8")
    total_samples = 0
    samples_with_images = 0
    empty_image_samples = 0
    unique_sequences = 0
    duplicate_sequences = 0
    shard_counts = [0 for _ in shard_paths]
    try:
        with JsonArrayWriter(qa_output_path) as qa_writer:
            iterator = iter_json_array(
                qa_input_path,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
            qa_bar = tqdm(
                total=(None if max_samples is None else int(max_samples)),
                desc=f"{progress_desc} qa",
                unit="sample",
                **tqdm_options(disable=not show_progress),
            )
            if show_progress:
                qa_bar.refresh()
            try:
                for item in iterator:
                    if max_samples is not None and total_samples >= int(max_samples):
                        break

                    total_samples += 1
                    item_dict = dict(item) if isinstance(item, dict) else {"raw_item": item}
                    images = item_dict.get("images", [])
                    latent_names: list[str] = []
                    if isinstance(images, list) and images:
                        image_refs = [str(img_ref) for img_ref in images]
                        digest = hash_image_sequence(image_refs)
                        latent_name = digest_to_name.get(digest)
                        if latent_name is None:
                            latent_idx = unique_sequences
                            latent_name = f"{latent_idx:09d}.npz"
                            digest_to_name[digest] = latent_name
                            unique_sequences += 1
                            sequence_line = {
                                "latent_idx": latent_idx,
                                "latent_name": latent_name,
                                "sequence_hash": digest,
                                "num_views": len(image_refs),
                                "image_rel_paths": image_refs,
                            }
                            line_text = json.dumps(sequence_line, ensure_ascii=False)
                            main_manifest_file.write(line_text + "\n")
                            shard_idx = latent_idx % len(shard_files)
                            shard_files[shard_idx].write(line_text + "\n")
                            shard_counts[shard_idx] += 1
                        else:
                            duplicate_sequences += 1
                        latent_names = [latent_name]
                        samples_with_images += 1
                    else:
                        empty_image_samples += 1

                    item_dict["latents"] = latent_names
                    qa_writer.write(item_dict)

                    qa_bar.update(1)
                    qa_bar.set_postfix(
                        unique=unique_sequences,
                        duplicates=duplicate_sequences,
                        empty=empty_image_samples,
                        refresh=False,
                    )
            finally:
                qa_bar.close()
    finally:
        for shard_file in shard_files:
            shard_file.close()
        main_manifest_file.close()

    return {
        "qa_input_path": qa_input_path,
        "qa_output_path": qa_output_path,
        "sequence_manifest_path": str(sequence_manifest_path),
        "num_sequence_shards": len(shard_paths),
        "total_samples": total_samples,
        "samples_with_images": samples_with_images,
        "empty_image_samples": empty_image_samples,
        "unique_sequences": unique_sequences,
        "duplicate_sequences": duplicate_sequences,
        "shard_counts": shard_counts,
    }


def load_sample_records_from_sequence_manifest(
    manifest_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    latent_dir: str | os.PathLike[str],
    default_camera: dict[str, Any] | None = None,
) -> list[SampleRecord]:
    image_root = str(image_root)
    latent_dir = str(latent_dir)
    default_camera = dict(default_camera or {})
    records: list[SampleRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_rel_paths = tuple(str(path) for path in item.get("image_rel_paths", []))
            if not image_rel_paths:
                continue
            image_abs_paths = tuple(
                path if os.path.isabs(path) else os.path.join(image_root, path)
                for path in image_rel_paths
            )
            latent_idx = int(item.get("latent_idx", len(records)))
            latent_name = str(item.get("latent_name", f"{latent_idx:09d}.npz"))
            sample_key = Path(latent_name).stem
            records.append(
                SampleRecord(
                    metadata_index=latent_idx,
                    sample_key=sample_key,
                    image_rel_paths=image_rel_paths,
                    image_abs_paths=image_abs_paths,
                    poses=tuple((0.0, 0.0) for _ in image_rel_paths),
                    camera=dict(default_camera),
                    output_path=os.path.join(latent_dir, latent_name),
                )
            )
    return records


def write_manifest(
    manifest_path: str | os.PathLike[str],
    dataset_name: str,
    metadata_path: str | os.PathLike[str],
    camera_image_root: str | os.PathLike[str],
    records: Sequence[SampleRecord],
) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in records:
            line = {
                "dataset_name": dataset_name,
                "metadata_path": str(metadata_path),
                "camera_image_root": str(camera_image_root),
                "metadata_index": record.metadata_index,
                "sample_key": record.sample_key,
                "latent_path": record.output_path,
                "num_views": len(record.image_rel_paths),
                "image_rel_paths": list(record.image_rel_paths),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_extract_config(
    config_path: str | os.PathLike[str],
    config: dict[str, Any],
) -> None:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def select_inference_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _rotation_x(angle_rad: float) -> np.ndarray:
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float32,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    return np.asarray(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )


def _build_c2w_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(pitch_deg))
    rotation = _rotation_y(yaw_rad) @ _rotation_x(pitch_rad)
    extrinsic = np.zeros((3, 4), dtype=np.float32)
    extrinsic[:, :3] = rotation
    return extrinsic


def _camera_params_from_metadata(camera: dict[str, Any]) -> tuple[float, float, float, float, int, int]:
    width = int(round(float(camera.get("width", 512))))
    height = int(round(float(camera.get("height", 384))))

    fx = camera.get("fx")
    fy = camera.get("fy")
    cx = camera.get("cx")
    cy = camera.get("cy")
    if None not in (fx, fy, cx, cy):
        return (
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            width,
            height,
        )

    hfov_deg = float(camera.get("hfov_deg", 90.0))
    vfov_deg = float(camera.get("vfov_deg", hfov_deg * height / width))
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy, width, height


def build_camera_tokens(
    record: SampleRecord,
    processed_hw: tuple[int, int],
    cam_token_mode: str,
) -> torch.Tensor:
    num_views = len(record.image_rel_paths)
    if cam_token_mode == "zeros":
        return torch.zeros(1, num_views, 11, dtype=torch.float32)

    use_pose = cam_token_mode in {
        "metadata_pose_world_scale",
        "metadata_pose_zero_scale",
    }
    use_world_scale = cam_token_mode in {
        "metadata_pose_world_scale",
        "zeros_world_scale",
    }

    pose_tokens = torch.zeros(1, num_views, 9, dtype=torch.float32)
    if use_pose:
        fx, fy, cx, cy, src_w, src_h = _camera_params_from_metadata(record.camera)
        tgt_h, tgt_w = processed_hw
        scale_x = float(tgt_w) / float(src_w)
        scale_y = float(tgt_h) / float(src_h)
        fxfycxcy = np.asarray(
            [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y],
            dtype=np.float32,
        )
        extrinsics = np.stack(
            [_build_c2w_from_yaw_pitch(yaw, pitch) for yaw, pitch in record.poses],
            axis=0,
        )
        extrinsics = torch.from_numpy(extrinsics).unsqueeze(0)
        intrinsics = torch.from_numpy(
            np.repeat(fxfycxcy[None, :], num_views, axis=0)
        ).unsqueeze(0)
        pose_tokens = extri_intri_to_pose_encoding(
            extrinsics,
            intrinsics,
            image_size_hw=processed_hw,
        ).float()

    scale_tokens = torch.zeros(1, num_views, 2, dtype=torch.float32)
    if use_world_scale:
        scale_tokens[..., 1] = 1.0

    return torch.cat([pose_tokens, scale_tokens], dim=-1)


def compute_encoder_resize_hw(
    input_hw: tuple[int, int],
    vggt_imsize: int = 518,
    vggt_patch_size: int = 14,
) -> tuple[int, int]:
    input_h, input_w = input_hw
    if input_h > input_w:
        tgt_h = vggt_imsize
        tgt_w = (int(tgt_h * input_w / input_h) // vggt_patch_size) * vggt_patch_size
    else:
        tgt_w = vggt_imsize
        tgt_h = (int(tgt_w * input_h / input_w) // vggt_patch_size) * vggt_patch_size
    return int(tgt_h), int(tgt_w)


def load_reconstructor(
    checkpoint_path: str | os.PathLike[str],
    device: torch.device,
    attention_type: str = "bidirectional_cross_attention",
) -> torch.nn.Module:
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = EncDec_VitB8(
        pretrained_vggt=False,
        pretrained_patch_embed=False,
        attention_to_features_type=attention_type,
    )
    msg = model.load_state_dict(state_dict, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}"
        )

    reconstructor = model.reconstructor.to(device)
    reconstructor.eval()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reconstructor


def _quantize_embedding(embedding: np.ndarray, storage_dtype: str) -> dict[str, np.ndarray]:
    storage_dtype = str(storage_dtype).lower()
    if storage_dtype == "float32":
        return {"embedding": embedding.astype(np.float32)}
    if storage_dtype == "float16":
        return {"embedding": embedding.astype(np.float16)}
    if storage_dtype == "int8":
        max_abs = float(np.max(np.abs(embedding))) if embedding.size else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.clip(np.rint(embedding / scale), -127, 127).astype(np.int8)
        return {
            "embedding": quantized,
            "embedding_scale": np.asarray([scale], dtype=np.float32),
        }
    raise ValueError(f"unsupported storage_dtype: {storage_dtype}")


def _save_npz_atomic(output_path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(str(tmp_path), **payload)
    os.replace(tmp_path, output_path)


def _decode_scalar_from_npz(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def validate_output_npz(
    output_path: str | os.PathLike[str],
    record: SampleRecord,
) -> tuple[bool, str]:
    output_path = Path(output_path)
    if not output_path.is_file():
        return False, "missing_file"

    required_keys = {
        "embedding",
        "camera_tokens",
        "poses",
        "image_rel_paths",
        "processed_image_hw",
        "encoder_input_hw",
        "patch_grid_hw",
        "metadata_index",
        "num_views",
        "sample_key",
        "cam_token_mode",
        "storage_dtype",
    }

    try:
        with np.load(output_path, allow_pickle=False) as data:
            keys = set(data.files)
            missing_keys = sorted(required_keys - keys)
            if missing_keys:
                return False, f"missing_keys:{','.join(missing_keys)}"

            embedding = data["embedding"]
            camera_tokens = data["camera_tokens"]
            poses = data["poses"]
            image_rel_paths = data["image_rel_paths"]
            patch_grid_hw = data["patch_grid_hw"]
            metadata_index = int(_decode_scalar_from_npz(data["metadata_index"]))
            num_views = int(_decode_scalar_from_npz(data["num_views"]))
            sample_key = str(_decode_scalar_from_npz(data["sample_key"]))
            storage_dtype = str(_decode_scalar_from_npz(data["storage_dtype"]))

            if metadata_index != int(record.metadata_index):
                return False, "metadata_index_mismatch"
            if sample_key != str(record.sample_key):
                return False, "sample_key_mismatch"
            if num_views != len(record.image_rel_paths):
                return False, "num_views_mismatch"
            if embedding.ndim != 4:
                return False, "embedding_rank_invalid"
            if embedding.shape[0] != 1:
                return False, "embedding_batch_dim_invalid"
            if tuple(camera_tokens.shape) != (num_views, 11):
                return False, "camera_tokens_shape_invalid"
            if poses.ndim != 2 or poses.shape[0] != num_views:
                return False, "poses_shape_invalid"
            if image_rel_paths.shape[0] != num_views:
                return False, "image_rel_paths_shape_invalid"
            if embedding.shape[1] != num_views:
                return False, "embedding_view_dim_invalid"
            if embedding.shape[-1] != 768:
                return False, "embedding_channel_dim_invalid"
            if patch_grid_hw.shape[0] != 2:
                return False, "patch_grid_shape_invalid"
            if int(np.prod(patch_grid_hw)) != int(embedding.shape[2]):
                return False, "patch_grid_product_invalid"
            if storage_dtype == "int8" and "embedding_scale" not in keys:
                return False, "missing_embedding_scale"
    except Exception as exc:  # noqa: BLE001
        return False, f"load_failed:{type(exc).__name__}"

    return True, "ok"


def append_progress_line(
    progress_path: str | os.PathLike[str],
    record: SampleRecord,
    status: str,
    message: str | None = None,
) -> None:
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "timestamp": time.time(),
        "status": status,
        "metadata_index": int(record.metadata_index),
        "sample_key": record.sample_key,
        "output_path": record.output_path,
    }
    if message:
        line["message"] = message
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _scan_records_chunk(
    records: Sequence[SampleRecord],
    overwrite: bool,
) -> list[ScanResult]:
    results: list[ScanResult] = []
    for record in records:
        output_path = Path(record.output_path)
        if overwrite:
            if not output_path.is_file():
                results.append(
                    ScanResult(
                        metadata_index=record.metadata_index,
                        status="pending_missing",
                        reason="missing_file",
                    )
                )
                continue
            is_valid, reason = validate_output_npz(output_path, record)
            results.append(
                ScanResult(
                    metadata_index=record.metadata_index,
                    status="pending_overwrite_existing_valid" if is_valid else "pending_overwrite_existing_invalid",
                    reason=reason,
                )
            )
            continue

        if not output_path.is_file():
            results.append(
                ScanResult(
                    metadata_index=record.metadata_index,
                    status="pending_missing",
                    reason="missing_file",
                )
            )
            continue

        is_valid, reason = validate_output_npz(output_path, record)
        results.append(
            ScanResult(
                metadata_index=record.metadata_index,
                status="skipped_existing" if is_valid else "pending_invalid_existing",
                reason=reason,
            )
        )
    return results


def scan_records_multiprocess(
    records: Sequence[SampleRecord],
    overwrite: bool,
    num_workers: int,
    show_progress: bool = True,
    progress_desc: str = "Scanning outputs",
) -> tuple[list[SampleRecord], dict[str, Any]]:
    record_list = list(records)
    total_records = len(record_list)
    if total_records == 0:
        return [], {
            "total_records": 0,
            "pending_records": 0,
            "skipped_existing": 0,
            "pending_invalid_existing": 0,
            "pending_missing": 0,
            "pending_overwrite_existing_valid": 0,
            "pending_overwrite_existing_invalid": 0,
            "scan_workers": 0,
        }

    workers = max(1, min(int(num_workers), total_records))
    record_map = {record.metadata_index: record for record in record_list}

    if workers == 1:
        scan_results = _scan_records_chunk(record_list, overwrite=overwrite)
    else:
        chunk_size = max(1, total_records // (workers * 8))
        chunks = [
            record_list[i : i + chunk_size]
            for i in range(0, total_records, chunk_size)
        ]
        scan_results = []
        max_pending_futures = max(32, workers * 4)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = set()
            future_to_chunk: dict[Any, int] = {}
            submit_bar = tqdm(
                total=len(chunks),
                desc=f"{progress_desc} submit",
                unit="chunk",
                **tqdm_options(disable=not show_progress),
            )
            done_bar = tqdm(
                total=len(chunks),
                desc=progress_desc,
                unit="chunk",
                **tqdm_options(disable=not show_progress),
            )
            if show_progress:
                submit_bar.refresh()
                done_bar.refresh()
            try:
                for chunk_idx, chunk in enumerate(chunks):
                    if len(pending) >= max_pending_futures:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            _ = future_to_chunk.pop(future)
                            scan_results.extend(future.result())
                            done_bar.update(1)

                    future = ex.submit(_scan_records_chunk, chunk, overwrite)
                    pending.add(future)
                    future_to_chunk[future] = chunk_idx
                    submit_bar.update(1)
                    submit_bar.set_postfix(
                        inflight=len(pending),
                        max_pending=max_pending_futures,
                        refresh=False,
                    )

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        _ = future_to_chunk.pop(future)
                        scan_results.extend(future.result())
                        done_bar.update(1)
            finally:
                submit_bar.close()
                done_bar.close()

    counts = {
        "skipped_existing": 0,
        "pending_invalid_existing": 0,
        "pending_missing": 0,
        "pending_overwrite_existing_valid": 0,
        "pending_overwrite_existing_invalid": 0,
    }
    pending_records: list[SampleRecord] = []
    for result in scan_results:
        counts[result.status] = counts.get(result.status, 0) + 1
        record = record_map[result.metadata_index]
        if result.status == "skipped_existing":
            continue
        if result.status in {
            "pending_invalid_existing",
            "pending_overwrite_existing_invalid",
        }:
            try:
                Path(record.output_path).unlink()
            except OSError:
                pass
        pending_records.append(record)

    summary = {
        "total_records": total_records,
        "pending_records": len(pending_records),
        "scan_workers": workers,
        **counts,
    }
    return pending_records, summary


def extract_single_record(
    reconstructor: torch.nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    record: SampleRecord,
    input_mode: str,
    target_size: int,
    cam_token_mode: str,
    storage_dtype: str,
) -> dict[str, Any]:
    images = load_and_preprocess_images(
        list(record.image_abs_paths),
        mode=input_mode,
        target_size=target_size,
        patch_size=8,
    )
    processed_hw = (int(images.shape[-2]), int(images.shape[-1]))
    camera_tokens = build_camera_tokens(record, processed_hw, cam_token_mode)

    images = images.unsqueeze(0).to(device=device, non_blocking=True)
    camera_tokens = camera_tokens.to(device=device, non_blocking=True)
    if device.type == "cuda":
        images = images.to(dtype=amp_dtype)
        camera_tokens = camera_tokens.to(dtype=amp_dtype)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            embedding = reconstructor(images, camera_tokens)

    embedding = embedding.detach().float().cpu().numpy()
    camera_tokens_np = camera_tokens.squeeze(0).detach().float().cpu().numpy()
    poses_np = np.asarray(record.poses, dtype=np.float32)

    encoder_input_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = np.asarray(
        [encoder_input_hw[0] // 14, encoder_input_hw[1] // 14],
        dtype=np.int32,
    )
    if int(np.prod(patch_grid_hw)) != int(embedding.shape[2]):
        raise RuntimeError(
            f"patch grid mismatch for {record.sample_key}: "
            f"expected {int(np.prod(patch_grid_hw))}, got {embedding.shape[2]}"
        )

    payload: dict[str, Any] = {
        **_quantize_embedding(embedding, storage_dtype),
        "camera_tokens": camera_tokens_np.astype(np.float16),
        "poses": poses_np.astype(np.float32),
        "image_rel_paths": np.asarray(record.image_rel_paths),
        "processed_image_hw": np.asarray(processed_hw, dtype=np.int32),
        "encoder_input_hw": np.asarray(encoder_input_hw, dtype=np.int32),
        "patch_grid_hw": patch_grid_hw,
        "metadata_index": np.asarray([record.metadata_index], dtype=np.int32),
        "num_views": np.asarray([len(record.image_rel_paths)], dtype=np.int32),
        "sample_key": np.asarray(record.sample_key),
        "cam_token_mode": np.asarray(cam_token_mode),
        "storage_dtype": np.asarray(storage_dtype),
    }
    return payload


def write_error_line(
    error_path: str | os.PathLike[str],
    record: SampleRecord,
    exc: BaseException,
) -> None:
    error_path = Path(error_path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "metadata_index": record.metadata_index,
        "sample_key": record.sample_key,
        "output_path": record.output_path,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with open(error_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def process_records(
    records: Sequence[SampleRecord],
    checkpoint_path: str | os.PathLike[str],
    device_str: str,
    attention_type: str,
    input_mode: str,
    target_size: int,
    cam_token_mode: str,
    storage_dtype: str,
    overwrite: bool,
    error_path: str | os.PathLike[str],
    progress_path: str | os.PathLike[str],
    show_progress: bool = True,
    progress_desc: str = "Extracting LagerNVS embeddings",
    progress_queue: Any = None,
    progress_queue_label: str | None = None,
    log_every: int = 50,
) -> dict[str, Any]:
    if len(records) == 0:
        return {
            "device": device_str,
            "attention_type": attention_type,
            "input_mode": input_mode,
            "target_size": int(target_size),
            "cam_token_mode": cam_token_mode,
            "storage_dtype": storage_dtype,
            "total_records": 0,
            "processed": 0,
            "skipped_existing": 0,
            "regenerated_invalid_existing": 0,
            "failed": 0,
            "elapsed_sec": 0.0,
        }

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = select_inference_dtype(device)
    model_load_started_at = time.time()
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "worker_status",
                "label": progress_queue_label,
                "status": "loading_model",
                "device": device_str,
                "total_records": int(len(records)),
            }
        )
    reconstructor = load_reconstructor(
        checkpoint_path=checkpoint_path,
        device=device,
        attention_type=attention_type,
    )
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "worker_status",
                "label": progress_queue_label,
                "status": "model_ready",
                "device": device_str,
                "model_load_sec": round(time.time() - model_load_started_at, 3),
                "total_records": int(len(records)),
            }
        )

    started_at = time.time()
    summary = {
        "device": device_str,
        "attention_type": attention_type,
        "input_mode": input_mode,
        "target_size": int(target_size),
        "cam_token_mode": cam_token_mode,
        "storage_dtype": storage_dtype,
        "total_records": int(len(records)),
        "processed": 0,
        "skipped_existing": 0,
        "regenerated_invalid_existing": 0,
        "failed": 0,
    }

    iterator = tqdm(
        records,
        desc=progress_desc,
        unit="sample",
        **tqdm_options(disable=not show_progress),
    )
    if show_progress:
        iterator.refresh()
    for record in iterator:
        output_path = Path(record.output_path)
        if output_path.is_file():
            if overwrite:
                pass
            else:
                is_valid, reason = validate_output_npz(output_path, record)
                if is_valid:
                    summary["skipped_existing"] += 1
                    append_progress_line(
                        progress_path=progress_path,
                        record=record,
                        status="skipped_existing",
                    )
                    if progress_queue is not None:
                        progress_queue.put(
                            {
                                "kind": "sample",
                                "label": progress_queue_label,
                                "status": "skipped_existing",
                                "metadata_index": record.metadata_index,
                            }
                        )
                    continue
                try:
                    output_path.unlink()
                except OSError:
                    pass
                summary["regenerated_invalid_existing"] += 1
                append_progress_line(
                    progress_path=progress_path,
                    record=record,
                    status="regenerate_invalid_existing",
                    message=reason,
                )
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "kind": "sample",
                            "label": progress_queue_label,
                            "status": "regenerate_invalid_existing",
                            "metadata_index": record.metadata_index,
                        }
                    )

        try:
            payload = extract_single_record(
                reconstructor=reconstructor,
                device=device,
                amp_dtype=amp_dtype,
                record=record,
                input_mode=input_mode,
                target_size=target_size,
                cam_token_mode=cam_token_mode,
                storage_dtype=storage_dtype,
            )
            _save_npz_atomic(output_path, payload)
            summary["processed"] += 1
            append_progress_line(
                progress_path=progress_path,
                record=record,
                status="processed",
            )
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "kind": "sample",
                        "label": progress_queue_label,
                        "status": "processed",
                        "metadata_index": record.metadata_index,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            write_error_line(error_path=error_path, record=record, exc=exc)
            append_progress_line(
                progress_path=progress_path,
                record=record,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "kind": "sample",
                        "label": progress_queue_label,
                        "status": "failed",
                        "metadata_index": record.metadata_index,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        completed = (
            summary["processed"]
            + summary["failed"]
            + summary["skipped_existing"]
        )
        remaining = max(0, len(records) - completed)
        if show_progress:
            iterator.set_postfix(
                processed=summary["processed"],
                failed=summary["failed"],
                remaining=remaining,
                refresh=False,
            )
        if log_every > 0 and completed > 0 and completed % int(log_every) == 0:
            log_info(
                f"{progress_desc}: processed={summary['processed']}, "
                f"failed={summary['failed']}, remaining={remaining}"
            )

    summary["elapsed_sec"] = round(time.time() - started_at, 3)
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "worker_done",
                "label": progress_queue_label,
                "summary": summary,
            }
        )
    return summary


def write_summary(summary_path: str | os.PathLike[str], summary: dict[str, Any]) -> None:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def shard_records(
    records: Sequence[SampleRecord],
    num_shards: int,
) -> list[list[SampleRecord]]:
    num_shards = max(1, int(num_shards))
    shards: list[list[SampleRecord]] = [[] for _ in range(num_shards)]
    for idx, record in enumerate(records):
        shards[idx % num_shards].append(record)
    return shards
