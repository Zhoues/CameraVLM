"""
/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp_gt_pose.py \
  --dataset_name hstar_sft_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1 \
  --workers_per_gpu 5

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp_gt_pose.py \
  --dataset_name hstar_sft_ours_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 5

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp_gt_pose.py \
  --dataset_name PAP_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 6

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp_gt_pose.py \
  --dataset_name raw_pano_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 6 \
  --num_machines 2 \
  --machine_rank 0

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp_gt_pose.py \
  --dataset_name raw_pano_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 6 \
  --num_machines 2 \
  --machine_rank 1 \

"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from queue import Empty
from typing import Any

import torch
from tqdm import tqdm

from lagernvs_feature_utils import (
    DEFAULT_CAMERA_DATA_ROOT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LATENT_ROOT,
    JsonArrayWriter,
    QALatentPaths,
    SampleRecord,
    build_qa_latent_manifest,
    compute_encoder_resize_hw,
    configure_stdout_for_tqdm,
    ensure_output_dirs,
    iter_json_array,
    load_sample_records_from_sequence_manifest,
    log_info,
    process_records,
    resolve_qa_latent_paths,
    tqdm_options,
    write_extract_config,
    write_summary,
)


CAM_TOKEN_CHOICES = (
    "zeros",
    "zeros_world_scale",
    "metadata_pose_world_scale",
    "metadata_pose_zero_scale",
)
STORAGE_DTYPE_CHOICES = ("float16", "int8", "float32")
DEFAULT_CAMERA = {
    "width": 512,
    "height": 384,
    "hfov_deg": 90.0,
    "vfov_deg": 67.5,
}
QA6_OLD_PROMPT = "Your response should be in the format of: <think>..."
QA6_NEW_PROMPT = (
    "Your response should be in the format of: "
    "<world>...</world><think>..."
)
RAW_PANO_V2_DATASET_NAME = "raw_pano_512x384_fov_90_v2"


def _parse_gpu_ids(gpus: str) -> list[int]:
    text = str(gpus or "").strip()
    if not text:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _infer_processed_hw(input_mode: str, target_size: int) -> tuple[int, int]:
    if input_mode == "square_crop":
        return int(target_size), int(target_size)
    width = int(DEFAULT_CAMERA["width"])
    height = int(DEFAULT_CAMERA["height"])
    if width >= height:
        new_width = int(target_size)
        new_height = int(round(height * (target_size / width) / 8) * 8)
    else:
        new_height = int(target_size)
        new_width = int(round(width * (target_size / height) / 8) * 8)
    return new_height, new_width


def _worker_sequence_manifest_path(
    sequence_manifest_dir: str | os.PathLike[str],
    worker_rank: int,
) -> str:
    return str(Path(sequence_manifest_dir) / f"worker_{worker_rank:02d}.jsonl")


def _worker_main(
    machine_rank: int,
    num_machines: int,
    worker_rank: int,
    gpu_id: int,
    paths: QALatentPaths,
    local_sequence_manifest_dir: str,
    camera_image_root: str | None,
    checkpoint_path: str,
    attention_type: str,
    input_mode: str,
    target_size: int,
    cam_token_mode: str,
    use_gt_pose: bool,
    storage_dtype: str,
    overwrite: bool,
    startup_delay_sec: float,
    progress_queue: Any = None,
) -> None:
    configure_stdout_for_tqdm()
    _machine_worker_dir(paths, num_machines, machine_rank).mkdir(parents=True, exist_ok=True)
    if startup_delay_sec > 0:
        time.sleep(float(startup_delay_sec) * float(worker_rank))

    manifest_path = _worker_sequence_manifest_path(local_sequence_manifest_dir, worker_rank)
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "worker_status",
                "label": f"worker-{worker_rank}",
                "status": "loading_shard",
                "device": f"cuda:{gpu_id}",
                "manifest_path": manifest_path,
            }
        )
    if use_gt_pose:
        if not camera_image_root:
            raise ValueError("camera_image_root is required when use_gt_pose=True")
        records = _load_sample_records_from_sequence_manifest_with_gt_pose(
            manifest_path=manifest_path,
            image_root=paths.image_root,
            latent_dir=paths.latent_dir,
            camera_image_root=camera_image_root,
            default_camera=DEFAULT_CAMERA,
        )
    else:
        records = load_sample_records_from_sequence_manifest(
            manifest_path=manifest_path,
            image_root=paths.image_root,
            latent_dir=paths.latent_dir,
            default_camera=DEFAULT_CAMERA,
        )
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "worker_status",
                "label": f"worker-{worker_rank}",
                "status": "shard_ready",
                "device": f"cuda:{gpu_id}",
                "assigned_records": len(records),
            }
        )

    summary = process_records(
        records=records,
        checkpoint_path=checkpoint_path,
        device_str=f"cuda:{gpu_id}",
        attention_type=attention_type,
        input_mode=input_mode,
        target_size=target_size,
        cam_token_mode=cam_token_mode,
        storage_dtype=storage_dtype,
        overwrite=overwrite,
        error_path=_worker_error_path(paths, num_machines, machine_rank, worker_rank),
        progress_path=_worker_progress_path(paths, num_machines, machine_rank, worker_rank),
        show_progress=False,
        progress_desc=f"qa-worker-{worker_rank}",
        progress_queue=progress_queue,
        progress_queue_label=f"worker-{worker_rank}",
        log_every=0,
    )
    summary.update(
        {
            "worker_rank": int(worker_rank),
            "gpu_id": int(gpu_id),
            "assigned_records": len(records),
            "manifest_path": manifest_path,
        }
    )
    write_summary(
        _worker_summary_path(paths, num_machines, machine_rank, worker_rank),
        summary,
    )


def _aggregate_worker_summaries(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
    worker_specs: list[tuple[int, int]],
    process_exitcodes: list[int],
) -> dict[str, object]:
    workers: list[dict[str, object]] = []
    totals = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "total_records": 0,
        "regenerated_invalid_existing": 0,
    }
    for worker_rank, gpu_id in worker_specs:
        summary_path = Path(
            _worker_summary_path(paths, num_machines, machine_rank, worker_rank)
        )
        if summary_path.is_file():
            with open(summary_path, "r", encoding="utf-8") as f:
                worker_summary = json.load(f)
        else:
            worker_summary = {
                "worker_rank": int(worker_rank),
                "gpu_id": int(gpu_id),
                "processed": 0,
                "skipped_existing": 0,
                "failed": 0,
                "total_records": 0,
                "regenerated_invalid_existing": 0,
                "missing_summary": True,
            }
        worker_summary["exitcode"] = int(process_exitcodes[worker_rank])
        workers.append(worker_summary)
        for key in totals:
            totals[key] += int(worker_summary.get(key, 0))

    return {
        **totals,
        "workers": workers,
        "num_workers": len(worker_specs),
    }


def _machine_run_dir(
    paths: QALatentPaths,
    num_machines: int,
) -> Path:
    return Path(paths.latent_dir) / "_machine_runs" / f"{int(num_machines):02d}_machines"


def _machine_worker_dir(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
) -> Path:
    return _machine_run_dir(paths, num_machines) / f"machine_{int(machine_rank):02d}" / "workers"


def _machine_summary_path(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
) -> str:
    return str(
        _machine_run_dir(paths, num_machines)
        / f"machine_{int(machine_rank):02d}.summary.json"
    )


def _worker_summary_path(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
    worker_rank: int,
) -> str:
    return str(
        _machine_worker_dir(paths, num_machines, machine_rank)
        / f"qa_worker_{worker_rank:02d}.summary.json"
    )


def _worker_error_path(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
    worker_rank: int,
) -> str:
    return str(
        _machine_worker_dir(paths, num_machines, machine_rank)
        / f"qa_worker_{worker_rank:02d}.errors.jsonl"
    )


def _worker_progress_path(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
    worker_rank: int,
) -> str:
    return str(
        _machine_worker_dir(paths, num_machines, machine_rank)
        / f"qa_worker_{worker_rank:02d}.progress.jsonl"
    )


def _machine_config_path(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
) -> str:
    return str(
        _machine_run_dir(paths, num_machines)
        / f"machine_{int(machine_rank):02d}.config.json"
    )


def _distributed_ready_marker_path(
    paths: QALatentPaths,
    num_machines: int,
) -> str:
    return str(
        _machine_run_dir(paths, num_machines)
        / "distributed_manifest.ready.json"
    )


def _global_machine_shard_dir(paths: QALatentPaths, num_machines: int) -> str:
    return str(
        Path(paths.latent_dir) / "_qa_machine_shards" / f"{int(num_machines):02d}_machines"
    )


def _local_worker_shard_dir(
    paths: QALatentPaths,
    num_machines: int,
    machine_rank: int,
) -> str:
    return str(
        _machine_run_dir(paths, num_machines)
        / f"machine_{int(machine_rank):02d}"
        / "_qa_worker_shards"
    )


def _count_jsonl_records(path: str | os.PathLike[str]) -> int:
    file_path = Path(path)
    if not file_path.is_file():
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _reset_local_worker_manifest_dir(
    local_sequence_manifest_dir: str | os.PathLike[str],
    num_local_workers: int,
) -> list[str]:
    dir_path = Path(local_sequence_manifest_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    manifest_paths = []
    for worker_rank in range(max(1, int(num_local_workers))):
        manifest_path = dir_path / f"worker_{worker_rank:02d}.jsonl"
        try:
            manifest_path.unlink()
        except OSError:
            pass
        manifest_paths.append(str(manifest_path))
    return manifest_paths


def _split_machine_manifest_to_local_workers(
    machine_manifest_path: str | os.PathLike[str],
    local_sequence_manifest_dir: str | os.PathLike[str],
    num_local_workers: int,
) -> list[int]:
    manifest_paths = _reset_local_worker_manifest_dir(
        local_sequence_manifest_dir=local_sequence_manifest_dir,
        num_local_workers=num_local_workers,
    )
    counts = [0 for _ in manifest_paths]
    if not manifest_paths:
        return counts

    with open(machine_manifest_path, "r", encoding="utf-8") as f_in:
        writers = [open(path, "w", encoding="utf-8") for path in manifest_paths]
        try:
            item_idx = 0
            for line in f_in:
                if not line.strip():
                    continue
                target_idx = item_idx % len(writers)
                writers[target_idx].write(line)
                counts[target_idx] += 1
                item_idx += 1
        finally:
            for writer in writers:
                writer.close()
    return counts


def _write_distributed_ready_marker(
    marker_path: str | os.PathLike[str],
    payload: dict[str, Any],
) -> None:
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = marker_path.with_name(marker_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, marker_path)


def _wait_for_distributed_ready_marker(
    marker_path: str | os.PathLike[str],
    timeout_sec: float,
    poll_sec: float,
    log_interval_sec: float = 30.0,
) -> dict[str, Any]:
    marker_path = Path(marker_path)
    started_at = time.time()
    last_log_at = 0.0
    log_info(
        f"waiting for distributed ready marker: {marker_path} "
        f"(timeout_sec={float(timeout_sec):.0f}, poll_sec={float(poll_sec):.1f})"
    )
    while True:
        if marker_path.is_file():
            elapsed_sec = time.time() - started_at
            log_info(
                f"distributed ready marker detected after {elapsed_sec:.1f}s: {marker_path}"
            )
            with open(marker_path, "r", encoding="utf-8") as f:
                return json.load(f)
        now = time.time()
        elapsed_sec = now - started_at
        if elapsed_sec > float(timeout_sec):
            raise TimeoutError(f"timed out waiting for distributed ready marker: {marker_path}")
        if now - last_log_at >= float(log_interval_sec):
            log_info(
                f"still waiting for distributed ready marker: {marker_path} "
                f"(elapsed_sec={elapsed_sec:.1f})"
            )
            last_log_at = now
        time.sleep(float(poll_sec))


def _replace_human_prompt_for_qa6(item: dict[str, Any]) -> dict[str, Any]:
    item_dict = dict(item)
    conversations = item_dict.get("conversations")
    if not isinstance(conversations, list):
        return item_dict

    rewritten = []
    for conversation in conversations:
        if not isinstance(conversation, dict):
            rewritten.append(conversation)
            continue
        conversation_dict = dict(conversation)
        if (
            conversation_dict.get("from") == "human"
            and isinstance(conversation_dict.get("value"), str)
        ):
            conversation_dict["value"] = conversation_dict["value"].replace(
                QA6_OLD_PROMPT,
                QA6_NEW_PROMPT,
            )
        rewritten.append(conversation_dict)
    item_dict["conversations"] = rewritten
    return item_dict


def _derive_qa_v6_output_path(qa_output_path: str) -> str:
    qa_path = Path(qa_output_path)
    name = qa_path.name
    if "QA_v5_w_latent" in name:
        return str(qa_path.with_name(name.replace("QA_v5_w_latent", "QA_v6_w_latent_prompt")))
    if "QA_v4_w_thinking" in name:
        return str(qa_path.with_name(name.replace("QA_v4_w_thinking", "QA_v6_w_latent_prompt")))
    return str(qa_path.with_name(f"{qa_path.stem}_v6_prompt{qa_path.suffix}"))


def _build_qa_v6_latent_prompt(
    qa_input_path: str | os.PathLike[str],
    qa_v6_output_path: str | os.PathLike[str],
    show_progress: bool,
) -> dict[str, Any]:
    input_path = str(qa_input_path)
    output_path = str(qa_v6_output_path)
    total_items = 0
    modified_items = 0
    with JsonArrayWriter(output_path) as writer:
        for item in iter_json_array(
            input_path,
            show_progress=show_progress,
            progress_desc=f"{Path(output_path).name} rewrite",
        ):
            total_items += 1
            rewritten = _replace_human_prompt_for_qa6(item if isinstance(item, dict) else {"raw_item": item})
            if rewritten != item:
                modified_items += 1
            writer.write(rewritten)
    return {
        "qa_input_path": input_path,
        "qa_v6_output_path": output_path,
        "total_items": total_items,
        "modified_items": modified_items,
    }


def _merge_machine_summaries(
    paths: QALatentPaths,
    num_machines: int,
    qa_scan_summary: dict[str, Any],
    sequence_manifest_dir: str | os.PathLike[str] | None = None,
    scan_workers: int = 1,
    scan_log_every: int = 100000,
    show_progress: bool = True,
) -> dict[str, Any]:
    log_info(
        f"merging machine summaries for dataset={paths.dataset_name}, "
        f"num_machines={int(num_machines)}"
    )
    machine_summaries = []
    totals = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "local_total_records": 0,
        "regenerated_invalid_existing": 0,
    }
    for machine_rank in range(int(num_machines)):
        summary_path = Path(_machine_summary_path(paths, num_machines, machine_rank))
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing machine summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        machine_summaries.append(summary)
        for key in totals:
            totals[key] += int(summary.get(key, 0))

    expected_latents = 0
    existing_latents = 0
    missing_latent_examples: list[str] = []
    expected_total = int(qa_scan_summary.get("unique_sequences", 0))
    requested_scan_workers = max(1, int(scan_workers))
    shard_dir = Path(sequence_manifest_dir) if sequence_manifest_dir is not None else None
    shard_paths: list[Path] = []
    if shard_dir is not None and shard_dir.is_dir():
        shard_paths = sorted(shard_dir.glob("worker_*.jsonl"))

    if shard_paths:
        actual_scan_workers = min(requested_scan_workers, len(shard_paths))
        log_info(
            f"starting parallel latent verification from {len(shard_paths)} manifest shards "
            f"(scan_workers={actual_scan_workers}, expected_latents={expected_total})"
        )
        progress_bar = tqdm(
            total=(expected_total if expected_total > 0 else len(shard_paths)),
            desc=f"{paths.dataset_name} merge verify",
            unit=("sample" if expected_total > 0 else "shard"),
            **tqdm_options(disable=not show_progress),
        )
        try:
            if actual_scan_workers == 1:
                for shard_path in shard_paths:
                    result = _verify_latent_manifest_shard(
                        (
                            str(shard_path),
                            paths.latent_dir,
                            20,
                            scan_log_every,
                        )
                    )
                    expected_latents += int(result["expected_latents"])
                    existing_latents += int(result["existing_latents"])
                    missing_latent_examples.extend(
                        result["missing_latent_examples"][
                            : max(0, 20 - len(missing_latent_examples))
                        ]
                    )
                    progress_bar.update(
                        int(result["expected_latents"]) if expected_total > 0 else 1
                    )
                    progress_bar.set_postfix(
                        expected=expected_latents,
                        existing=existing_latents,
                        missing=max(0, expected_latents - existing_latents),
                        refresh=True,
                    )
                    log_info(
                        f"verified shard {Path(result['manifest_path']).name}: "
                        f"expected={result['expected_latents']}, "
                        f"existing={result['existing_latents']}, "
                        f"missing={result['missing_latent_count']}"
                    )
            else:
                tasks = [
                    (
                        str(shard_path),
                        paths.latent_dir,
                        20,
                        scan_log_every,
                    )
                    for shard_path in shard_paths
                ]
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=actual_scan_workers) as pool:
                    for result in pool.imap_unordered(_verify_latent_manifest_shard, tasks):
                        expected_latents += int(result["expected_latents"])
                        existing_latents += int(result["existing_latents"])
                        missing_latent_examples.extend(
                            result["missing_latent_examples"][
                                : max(0, 20 - len(missing_latent_examples))
                            ]
                        )
                        progress_bar.update(
                            int(result["expected_latents"]) if expected_total > 0 else 1
                        )
                        progress_bar.set_postfix(
                            expected=expected_latents,
                            existing=existing_latents,
                            missing=max(0, expected_latents - existing_latents),
                            refresh=True,
                        )
                        log_info(
                            f"verified shard {Path(result['manifest_path']).name}: "
                            f"expected={result['expected_latents']}, "
                            f"existing={result['existing_latents']}, "
                            f"missing={result['missing_latent_count']}"
                        )
        finally:
            progress_bar.set_postfix(
                expected=expected_latents,
                existing=existing_latents,
                missing=max(0, expected_latents - existing_latents),
                refresh=True,
            )
            progress_bar.close()
    else:
        manifest_path = Path(paths.sequence_manifest_path)
        log_info(
            f"manifest shard directory unavailable, fallback to single-file verification: "
            f"{manifest_path}"
        )
        result = _verify_latent_manifest_shard(
            (
                str(manifest_path),
                paths.latent_dir,
                20,
                scan_log_every,
            )
        )
        expected_latents = int(result["expected_latents"])
        existing_latents = int(result["existing_latents"])
        missing_latent_examples = list(result["missing_latent_examples"])

    merged_summary = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "sequence_manifest_path": paths.sequence_manifest_path,
        "latent_dir": paths.latent_dir,
        "num_machines": int(num_machines),
        "qa_scan": qa_scan_summary,
        "expected_latents": expected_latents,
        "existing_latents": existing_latents,
        "missing_latent_count": max(0, expected_latents - existing_latents),
        "missing_latent_examples": missing_latent_examples,
        "machines": machine_summaries,
        **totals,
    }
    log_info(
        f"merge verification finished: expected_latents={expected_latents}, "
        f"existing_latents={existing_latents}, "
        f"missing_latent_count={merged_summary['missing_latent_count']}, "
        f"failed={merged_summary['failed']}"
    )
    return merged_summary


def _verify_latent_manifest_shard(
    args: tuple[str, str, int, int | None],
) -> dict[str, Any]:
    manifest_path_text, latent_dir_text, missing_example_limit, log_every = args
    manifest_path = Path(manifest_path_text)
    latent_dir = Path(latent_dir_text)
    expected_latents = 0
    existing_latents = 0
    missing_latent_examples: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            expected_latents += 1
            item = json.loads(line)
            latent_name = str(item.get("latent_name", ""))
            latent_path = latent_dir / latent_name
            if latent_path.is_file():
                existing_latents += 1
            elif len(missing_latent_examples) < int(missing_example_limit):
                missing_latent_examples.append(str(latent_path))
            if (
                log_every is not None
                and int(log_every) > 0
                and expected_latents % int(log_every) == 0
            ):
                log_info(
                    f"verifying shard {manifest_path.name}: "
                    f"expected={expected_latents}, existing={existing_latents}, "
                    f"missing={max(0, expected_latents - existing_latents)}"
                )
    return {
        "manifest_path": str(manifest_path),
        "expected_latents": expected_latents,
        "existing_latents": existing_latents,
        "missing_latent_count": max(0, expected_latents - existing_latents),
        "missing_latent_examples": missing_latent_examples,
    }


def _resolve_annotation_path_with_fallback(
    paths: QALatentPaths,
    annotation_path_explicit: bool,
) -> QALatentPaths:
    annotation_path = Path(paths.annotation_path)
    if annotation_path.is_file():
        return paths
    if annotation_path_explicit:
        raise FileNotFoundError(f"annotation file not found: {annotation_path}")

    fallback_names = [
        "QA_v6_w_latent_prompt.json",
        "QA_v5_w_latent.json",
        "QA_v4_w_thinking.json",
    ]
    for fallback_name in fallback_names:
        fallback_path = annotation_path.with_name(fallback_name)
        if fallback_path.is_file():
            log_info(
                f"annotation_path {annotation_path} not found, fallback to {fallback_path}"
            )
            return replace(paths, annotation_path=str(fallback_path))
    raise FileNotFoundError(f"annotation file not found: {annotation_path}")


def _resolve_camera_image_root(
    dataset_name: str,
    image_root: str,
    camera_data_root: str,
    camera_image_root: str | None,
) -> str:
    if camera_image_root is not None:
        path = Path(camera_image_root)
        if not path.is_dir():
            raise FileNotFoundError(f"camera_image_root not found: {path}")
        return str(path)

    image_root_path = Path(image_root)
    candidates = []
    if image_root_path.name == "camera_image":
        candidates.append(image_root_path)
    candidates.append(Path(camera_data_root) / dataset_name / "camera_image")
    if dataset_name == "raw_pano_512x384_fov_90":
        candidates.append(Path(camera_data_root) / RAW_PANO_V2_DATASET_NAME / "camera_image")

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    prefix_matches = sorted(Path(camera_data_root).glob(f"{dataset_name}*/camera_image"))
    if len(prefix_matches) == 1 and prefix_matches[0].is_dir():
        return str(prefix_matches[0])

    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"failed to resolve camera_image_root for {dataset_name}; tried: {candidate_text}"
    )


def _apply_gt_output_suffix(
    paths: QALatentPaths,
    use_gt_pose: bool,
    qa_output_path_explicit: bool,
    separate_gt_pose_output: bool,
) -> QALatentPaths:
    if not use_gt_pose or not separate_gt_pose_output:
        return paths

    latent_dir = Path(f"{paths.latent_dir}__gt_pose")
    qa_output_path = Path(paths.qa_output_path)
    if qa_output_path_explicit:
        resolved_qa_output_path = qa_output_path
    else:
        resolved_qa_output_path = qa_output_path.with_name(
            f"{qa_output_path.stem}_gt_pose{qa_output_path.suffix}"
        )

    return replace(
        paths,
        latent_dir=str(latent_dir),
        sequence_manifest_dir=str(latent_dir / "_qa_sequence_shards"),
        sequence_manifest_path=str(latent_dir / "qa_sequence_manifest.jsonl"),
        qa_output_path=str(resolved_qa_output_path),
        config_path=str(latent_dir / "qa_extract_config.json"),
        summary_path=str(latent_dir / "qa_extract_summary.json"),
        error_dir=str(latent_dir / "_errors"),
        progress_dir=str(latent_dir / "_progress"),
    )


def _remove_tree(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {
            "path": str(path),
            "removed": False,
            "missing": True,
        }
    if not path.is_dir():
        raise NotADirectoryError(f"cleanup target is not a directory: {path}")

    started_at = time.time()
    shutil.rmtree(path)
    return {
        "path": str(path),
        "removed": True,
        "missing": False,
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _cleanup_distributed_artifacts(
    paths: QALatentPaths,
    num_machines: int,
    cleanup_workers: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    machine_run_dir = _machine_run_dir(paths, num_machines)
    machine_shard_dir = Path(_global_machine_shard_dir(paths, num_machines))
    targets = [
        str(machine_run_dir),
        str(machine_shard_dir),
    ]
    cleanup_workers = max(1, int(cleanup_workers))

    existing_targets = [target for target in targets if Path(target).exists()]
    results: list[dict[str, Any]] = []
    if existing_targets:
        log_info(
            f"starting cleanup of distributed artifacts: targets={len(existing_targets)}, "
            f"cleanup_workers={cleanup_workers}"
        )
        if cleanup_workers == 1 or len(existing_targets) == 1:
            progress_bar = tqdm(
                total=len(existing_targets),
                desc=f"{paths.dataset_name} cleanup",
                unit="dir",
                **tqdm_options(disable=(not show_progress)),
            )
            try:
                for target in existing_targets:
                    log_info(f"cleanup removing directory: {target}")
                    result = _remove_tree(target)
                    results.append(result)
                    progress_bar.update(1)
                    log_info(
                        f"cleanup removed directory: {target} "
                        f"(elapsed_sec={result.get('elapsed_sec', 'n/a')})"
                    )
            finally:
                progress_bar.close()
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(cleanup_workers, len(existing_targets))) as pool:
                progress_bar = tqdm(
                    total=len(existing_targets),
                    desc=f"{paths.dataset_name} cleanup",
                    unit="dir",
                    **tqdm_options(disable=(not show_progress)),
                )
                try:
                    for result in pool.imap_unordered(_remove_tree, existing_targets):
                        results.append(result)
                        progress_bar.update(1)
                        log_info(
                            f"cleanup removed directory: {result['path']} "
                            f"(elapsed_sec={result.get('elapsed_sec', 'n/a')})"
                        )
                finally:
                    progress_bar.close()
        log_info("distributed artifact cleanup finished")
    else:
        log_info("no distributed artifacts to clean up")

    parent_cleanup: list[str] = []
    for parent_dir in (machine_run_dir.parent, machine_shard_dir.parent):
        if parent_dir.is_dir():
            try:
                next(parent_dir.iterdir())
            except StopIteration:
                parent_dir.rmdir()
                parent_cleanup.append(str(parent_dir))

    return {
        "cleanup_workers": cleanup_workers,
        "targets": targets,
        "removed_targets": results,
        "removed_empty_parent_dirs": parent_cleanup,
    }


def _resolve_effective_cam_token_mode(
    cam_token_mode: str,
    use_gt_pose: bool,
) -> str:
    if not use_gt_pose:
        return cam_token_mode
    if cam_token_mode in {"zeros", "metadata_pose_zero_scale"}:
        return "metadata_pose_zero_scale"
    if cam_token_mode in {"zeros_world_scale", "metadata_pose_world_scale"}:
        return "metadata_pose_world_scale"
    raise ValueError(f"unsupported cam_token_mode: {cam_token_mode}")


def _derive_sequence_dir_rel_path(image_rel_path: str) -> str:
    rel_path = Path(str(image_rel_path))
    if rel_path.parent != Path("."):
        return str(rel_path.parent)
    stem = rel_path.stem
    if "_" not in stem:
        raise ValueError(f"cannot infer sequence dir from flat image name: {image_rel_path}")
    return stem.rsplit("_", 1)[0]


def _derive_annotation_frame_name(image_rel_path: str) -> str:
    rel_path = Path(str(image_rel_path))
    if rel_path.parent != Path("."):
        return rel_path.name
    stem = rel_path.stem
    if "_" not in stem:
        raise ValueError(f"cannot infer frame name from flat image name: {image_rel_path}")
    frame_id = stem.rsplit("_", 1)[1]
    return f"{frame_id}{rel_path.suffix}"


def _derive_frame_index(image_rel_path: str) -> int:
    frame_name = _derive_annotation_frame_name(image_rel_path)
    return int(Path(frame_name).stem)


def _extract_pose_from_annotation_frame(
    frame: dict[str, Any],
    annotation_path: Path,
    frame_name: str,
) -> tuple[float, float]:
    pose = frame.get("pose")
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    if "yaw" in frame and "pitch" in frame:
        return float(frame["yaw"]), float(frame["pitch"])
    raise KeyError(f"pose missing for frame {frame_name} in {annotation_path}")


def _normalize_yaw_deg(yaw_deg: float) -> float:
    return float(yaw_deg) % 360.0


def _load_sequence_metadata(
    annotation_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, tuple[float, float]]]:
    path = Path(annotation_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta_path = path.with_name("meta.json")
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)
        for key, value in meta_payload.items():
            if key not in payload:
                payload[key] = value

    frames = payload.get("frames", [])
    frame_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_image = frame.get("image")
            if frame_image:
                frame_lookup[Path(str(frame_image)).name] = frame

    image_pose_lookup: dict[str, tuple[float, float]] = {}
    images = payload.get("images", [])
    poses = payload.get("poses", [])
    if isinstance(images, list) and isinstance(poses, list):
        for image_name, pose in zip(images, poses):
            if not isinstance(pose, (list, tuple)) or len(pose) < 2:
                continue
            image_pose_lookup[Path(str(image_name)).name] = (
                float(pose[0]),
                float(pose[1]),
            )

    return payload, frame_lookup, image_pose_lookup


def _resolve_sequence_dir_path(
    image_rel_paths: tuple[str, ...],
    camera_image_root: Path,
    resolved_dir_cache: dict[str, Path],
) -> Path:
    raw_sequence_dirs = {_derive_sequence_dir_rel_path(path) for path in image_rel_paths}
    if len(raw_sequence_dirs) != 1:
        raise ValueError(
            f"multiple sequence dirs found in one manifest record: {sorted(raw_sequence_dirs)}"
        )
    raw_sequence_dir = next(iter(raw_sequence_dirs))

    cached = resolved_dir_cache.get(raw_sequence_dir)
    if cached is not None:
        return cached

    direct_path = camera_image_root / raw_sequence_dir
    if direct_path.is_dir():
        resolved_dir_cache[raw_sequence_dir] = direct_path
        return direct_path

    prefix_matches = sorted(
        candidate
        for candidate in camera_image_root.glob(f"{raw_sequence_dir}*")
        if candidate.is_dir()
    )
    if len(prefix_matches) == 1:
        resolved_dir_cache[raw_sequence_dir] = prefix_matches[0]
        return prefix_matches[0]

    raise FileNotFoundError(
        f"failed to resolve sequence dir for {raw_sequence_dir} under {camera_image_root}"
    )


def _resolve_pose_for_image(
    image_rel_path: str,
    payload: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
    image_pose_lookup: dict[str, tuple[float, float]],
    annotation_path: Path,
) -> tuple[float, float]:
    frame_name = _derive_annotation_frame_name(image_rel_path)

    frame = frame_lookup.get(frame_name)
    if frame is not None:
        return _extract_pose_from_annotation_frame(
            frame=frame,
            annotation_path=annotation_path,
            frame_name=frame_name,
        )

    pose = image_pose_lookup.get(frame_name)
    if pose is not None:
        return pose

    poses = payload.get("poses", [])
    frame_index = _derive_frame_index(image_rel_path)
    if isinstance(poses, list) and 1 <= frame_index <= len(poses):
        pose_item = poses[frame_index - 1]
        if isinstance(pose_item, (list, tuple)) and len(pose_item) >= 2:
            return float(pose_item[0]), float(pose_item[1])

    initial_yaw = payload.get("initial_yaw")
    initial_pitch = payload.get("initial_pitch")
    actions = payload.get("actions", [])
    if initial_yaw is not None and initial_pitch is not None and isinstance(actions, list):
        yaw = float(initial_yaw)
        pitch = float(initial_pitch)
        for action in actions[:frame_index]:
            if not isinstance(action, (list, tuple)) or len(action) < 2:
                continue
            yaw += float(action[0])
            pitch += float(action[1])
        return _normalize_yaw_deg(yaw), float(pitch)

    raise KeyError(f"pose not found for {frame_name} in {annotation_path}")


def _load_sample_records_from_sequence_manifest_with_gt_pose(
    manifest_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    latent_dir: str | os.PathLike[str],
    camera_image_root: str | os.PathLike[str],
    default_camera: dict[str, Any] | None = None,
) -> list[SampleRecord]:
    image_root = str(image_root)
    latent_dir = str(latent_dir)
    camera_image_root = Path(camera_image_root)
    default_camera = dict(default_camera or {})
    annotation_cache: dict[
        str,
        tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, tuple[float, float]]],
    ] = {}
    resolved_dir_cache: dict[str, Path] = {}
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

            sequence_dir_path = _resolve_sequence_dir_path(
                image_rel_paths=image_rel_paths,
                camera_image_root=camera_image_root,
                resolved_dir_cache=resolved_dir_cache,
            )
            annotation_path = sequence_dir_path / "annotation.json"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"annotation.json not found: {annotation_path}")

            annotation_key = str(annotation_path)
            annotation_payload, frame_lookup, image_pose_lookup = annotation_cache.get(
                annotation_key,
                ({}, {}, {}),
            )
            if not annotation_payload:
                annotation_payload, frame_lookup, image_pose_lookup = _load_sequence_metadata(
                    annotation_path
                )
                annotation_cache[annotation_key] = (
                    annotation_payload,
                    frame_lookup,
                    image_pose_lookup,
                )

            poses: list[tuple[float, float]] = []
            for image_rel_path in image_rel_paths:
                poses.append(
                    _resolve_pose_for_image(
                        image_rel_path=image_rel_path,
                        payload=annotation_payload,
                        frame_lookup=frame_lookup,
                        image_pose_lookup=image_pose_lookup,
                        annotation_path=annotation_path,
                    )
                )

            image_abs_paths = tuple(
                path if os.path.isabs(path) else os.path.join(image_root, path)
                for path in image_rel_paths
            )
            latent_idx = int(item.get("latent_idx", len(records)))
            latent_name = str(item.get("latent_name", f"{latent_idx:09d}.npz"))
            sample_key = Path(latent_name).stem
            camera = annotation_payload.get("camera", {})
            if not isinstance(camera, dict):
                camera = dict(default_camera)
            else:
                camera = {**default_camera, **camera}

            records.append(
                SampleRecord(
                    metadata_index=latent_idx,
                    sample_key=sample_key,
                    image_rel_paths=image_rel_paths,
                    image_abs_paths=image_abs_paths,
                    poses=tuple(poses),
                    camera=camera,
                    output_path=os.path.join(latent_dir, latent_name),
                )
            )

    return records


def main() -> None:
    configure_stdout_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Extract LagerNVS embeddings from unique QA image sequences with multiple GPUs/processes"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PAP_512x384_fov_90",
        help="数据集名称；会从 qwenvl.data 解析 annotation_path 和 image_root",
    )
    parser.add_argument("--annotation_path", type=str, default=None, help="可选；直接指定 QA_v4_w_thinking.json")
    parser.add_argument("--image_root", type=str, default=None, help="可选；覆盖图片根目录")
    parser.add_argument(
        "--camera_data_root",
        type=str,
        default=str(DEFAULT_CAMERA_DATA_ROOT),
        help="camera_data 根目录；use_gt_pose 时用于定位 annotation.json",
    )
    parser.add_argument(
        "--camera_image_root",
        type=str,
        default=None,
        help="可选；直接指定 camera_image 根目录；use_gt_pose 时优先使用",
    )
    parser.add_argument("--qa_output_path", type=str, default=None, help="可选；覆盖 QA_v5_w_latent.json 输出路径")
    parser.add_argument(
        "--qa_v6_output_path",
        type=str,
        default=None,
        help="可选；覆盖 QA_v6_w_latent_prompt.json 输出路径",
    )
    parser.add_argument("--latent_root", type=str, default=str(DEFAULT_LATENT_ROOT), help="latent_data 根目录")
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="checkpoint 路径")
    parser.add_argument("--num_machines", type=int, default=1, help="总机器数量")
    parser.add_argument("--machine_rank", type=int, default=0, help="当前机器 id，从 0 开始")
    parser.add_argument("--gpus", type=str, default="0", help="使用哪些 GPU，例如 0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="每张卡加载几个模型/起几个 worker")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=("bidirectional_cross_attention", "full_attention"),
        help="General 512 模型使用 bidirectional_cross_attention",
    )
    parser.add_argument("--input_mode", type=str, default="resize", choices=("resize", "square_crop"))
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--cam_token_mode", type=str, default="zeros", choices=CAM_TOKEN_CHOICES)
    parser.add_argument(
        "--use_gt_pose",
        action="store_true",
        help=(
            "使用 camera_data 里的 annotation.json 构造 GT yaw/pitch+intrinsics；"
            "会自动把 zeros/zeros_world_scale 提升为 metadata_pose_* 模式"
        ),
    )
    parser.add_argument(
        "--separate_gt_pose_output",
        action="store_true",
        help="把 GT pose 版输出写到独立的 __gt_pose latent/QA 路径；默认关闭，直接复用原数据集名字",
    )
    parser.add_argument("--storage_dtype", type=str, default="float16", choices=STORAGE_DTYPE_CHOICES)
    parser.add_argument("--max_samples", type=int, default=None, help="最多扫描多少条 QA，便于 debug")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 npz")
    parser.add_argument(
        "--manifest_wait_timeout_sec",
        type=float,
        default=7200.0,
        help="非 0 号机器等待全局 QA/manifest 就绪的最长时间",
    )
    parser.add_argument(
        "--manifest_poll_sec",
        type=float,
        default=10.0,
        help="等待全局 QA/manifest 时的轮询间隔",
    )
    parser.add_argument(
        "--reuse_manifest",
        dest="reuse_manifest",
        action="store_true",
        help="如果全局 manifest/QA 已存在则直接复用，适合分期重跑",
    )
    parser.add_argument(
        "--force_rebuild_manifest",
        dest="reuse_manifest",
        action="store_false",
        help="强制 0 号机器重建全局 manifest/QA",
    )
    parser.add_argument(
        "--generate_qa_v6",
        dest="generate_qa_v6",
        action="store_true",
        help="由 QA_v5_w_latent.json 同步生成 QA_v6_w_latent_prompt.json",
    )
    parser.add_argument(
        "--no_generate_qa_v6",
        dest="generate_qa_v6",
        action="store_false",
        help="不生成 QA_v6_w_latent_prompt.json",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="不做抽取，只合并所有机器 summary 并校验最终 latent 数量",
    )
    parser.add_argument(
        "--cleanup_after_merge",
        action="store_true",
        help="merge_only 校验通过后，删除分布式中间目录和各机器 summary/manifest",
    )
    parser.add_argument(
        "--cleanup_workers",
        type=int,
        default=8,
        help="cleanup_after_merge 时用于并行清理目录的进程数",
    )
    parser.add_argument(
        "--scan_workers",
        type=int,
        default=8,
        help="merge_only 校验 latent 是否齐全时用于并行扫描 manifest shard 的进程数",
    )
    parser.add_argument(
        "--scan_log_every",
        type=int,
        default=100000,
        help="merge_only 并行扫描时每处理多少条 manifest 记录打印一次子进程心跳日志；设为 0 则关闭",
    )
    parser.add_argument(
        "--startup_delay_sec",
        type=float,
        default=3.0,
        help="worker 启动间隔，避免同时加载 checkpoint 打爆 IO",
    )
    parser.add_argument("--no_progress", action="store_true", help="关闭 tqdm 进度条")
    parser.set_defaults(reuse_manifest=True, generate_qa_v6=True)
    args = parser.parse_args()

    if args.max_samples is not None:
        args.reuse_manifest = False

    num_machines = int(args.num_machines)
    machine_rank = int(args.machine_rank)
    if num_machines <= 0:
        raise ValueError("num_machines must be positive")
    if machine_rank < 0 or machine_rank >= num_machines:
        raise ValueError(
            f"machine_rank must be in [0, {num_machines - 1}], got {machine_rank}"
        )

    if args.merge_only:
        gpu_ids: list[int] = []
        worker_specs: list[tuple[int, int]] = []
    else:
        gpu_ids = _parse_gpu_ids(args.gpus)
        if not gpu_ids:
            raise RuntimeError("no GPU selected")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the multiprocessing extractor")

        worker_specs = []
        for gpu_id in gpu_ids:
            for _ in range(max(1, int(args.workers_per_gpu))):
                worker_specs.append((len(worker_specs), int(gpu_id)))
        if not worker_specs:
            raise RuntimeError("no worker specs generated")

    paths = resolve_qa_latent_paths(
        dataset_name=args.dataset_name,
        latent_root=args.latent_root,
        annotation_path=args.annotation_path,
        image_root=args.image_root,
        qa_output_path=args.qa_output_path,
    )
    paths = _resolve_annotation_path_with_fallback(
        paths=paths,
        annotation_path_explicit=(args.annotation_path is not None),
    )
    paths = _apply_gt_output_suffix(
        paths=paths,
        use_gt_pose=bool(args.use_gt_pose),
        qa_output_path_explicit=(args.qa_output_path is not None),
        separate_gt_pose_output=bool(args.separate_gt_pose_output),
    )
    effective_cam_token_mode = _resolve_effective_cam_token_mode(
        cam_token_mode=args.cam_token_mode,
        use_gt_pose=bool(args.use_gt_pose),
    )
    qa_v6_output_path = str(
        Path(args.qa_v6_output_path)
        if args.qa_v6_output_path is not None
        else Path(_derive_qa_v6_output_path(paths.qa_output_path))
    )
    resolved_camera_image_root = None
    if args.use_gt_pose:
        resolved_camera_image_root = _resolve_camera_image_root(
            dataset_name=args.dataset_name,
            image_root=paths.image_root,
            camera_data_root=args.camera_data_root,
            camera_image_root=args.camera_image_root,
        )

    global_sequence_manifest_dir = _global_machine_shard_dir(paths, num_machines)
    local_sequence_manifest_dir = _local_worker_shard_dir(
        paths=paths,
        num_machines=num_machines,
        machine_rank=machine_rank,
    )
    ready_marker_path = _distributed_ready_marker_path(paths, num_machines)

    ensure_output_dirs(paths)
    Path(paths.latent_dir, "_workers").mkdir(parents=True, exist_ok=True)
    Path(_machine_run_dir(paths, num_machines)).mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        ready_payload = _wait_for_distributed_ready_marker(
            marker_path=ready_marker_path,
            timeout_sec=args.manifest_wait_timeout_sec,
            poll_sec=args.manifest_poll_sec,
        )
        log_info("starting merge-only summary aggregation and latent verification")
        merged_summary = _merge_machine_summaries(
            paths=paths,
            num_machines=num_machines,
            qa_scan_summary=ready_payload["qa_scan_summary"],
            sequence_manifest_dir=ready_payload.get("sequence_manifest_dir"),
            scan_workers=args.scan_workers,
            scan_log_every=args.scan_log_every,
            show_progress=not args.no_progress,
        )
        merged_summary.update(
            {
                "use_gt_pose": bool(args.use_gt_pose),
                "separate_gt_pose_output": bool(args.separate_gt_pose_output),
                "qa_v6_output_path": ready_payload.get("qa_v6_output_path"),
                "ready_marker_path": ready_marker_path,
            }
        )
        cleanup_requested = bool(args.cleanup_after_merge)
        cleanup_reason = None
        cleanup_summary = None
        if cleanup_requested:
            if int(merged_summary.get("missing_latent_count", 0)) != 0:
                cleanup_reason = "missing_latents"
                log_info(
                    "skipping cleanup because merge verification found missing latents"
                )
            elif int(merged_summary.get("failed", 0)) != 0:
                cleanup_reason = "worker_failures"
                log_info("skipping cleanup because machine summaries report failures")
            else:
                cleanup_summary = _cleanup_distributed_artifacts(
                    paths=paths,
                    num_machines=num_machines,
                    cleanup_workers=args.cleanup_workers,
                    show_progress=not args.no_progress,
                )
        merged_summary["cleanup_requested"] = cleanup_requested
        merged_summary["cleanup_performed"] = cleanup_summary is not None
        merged_summary["cleanup_summary"] = cleanup_summary
        merged_summary["cleanup_skipped_reason"] = cleanup_reason
        log_info(f"writing merged summary to {paths.summary_path}")
        write_summary(paths.summary_path, merged_summary)
        print(json.dumps(merged_summary, ensure_ascii=False, indent=2), flush=True)
        return

    ready_payload: dict[str, Any]
    if machine_rank == 0:
        manifest_ready = Path(ready_marker_path).is_file()
        if args.reuse_manifest and manifest_ready:
            with open(ready_marker_path, "r", encoding="utf-8") as f:
                ready_payload = json.load(f)
            log_info(f"reusing distributed manifest from {ready_marker_path}")
            if args.generate_qa_v6 and not Path(qa_v6_output_path).is_file():
                qa_v6_summary = _build_qa_v6_latent_prompt(
                    qa_input_path=paths.qa_output_path,
                    qa_v6_output_path=qa_v6_output_path,
                    show_progress=not args.no_progress,
                )
                ready_payload["qa_v6_output_path"] = qa_v6_output_path
                ready_payload["qa_v6_summary"] = qa_v6_summary
                _write_distributed_ready_marker(ready_marker_path, ready_payload)
        else:
            log_info(f"loading QA annotation from {paths.annotation_path}")
            qa_scan_summary = build_qa_latent_manifest(
                qa_input_path=paths.annotation_path,
                qa_output_path=paths.qa_output_path,
                sequence_manifest_path=paths.sequence_manifest_path,
                sequence_manifest_dir=global_sequence_manifest_dir,
                num_sequence_shards=num_machines,
                show_progress=not args.no_progress,
                progress_desc=f"{paths.dataset_name} QA scan",
                max_samples=args.max_samples,
            )
            log_info(
                f"QA scan finished: total_samples={qa_scan_summary['total_samples']}, "
                f"unique_sequences={qa_scan_summary['unique_sequences']}, "
                f"duplicates={qa_scan_summary['duplicate_sequences']}, "
                f"empty_image_samples={qa_scan_summary['empty_image_samples']}"
            )
            if qa_scan_summary["unique_sequences"] <= 0:
                raise RuntimeError("no valid image sequences found in QA annotation")

            qa_v6_summary = None
            if args.generate_qa_v6:
                qa_v6_summary = _build_qa_v6_latent_prompt(
                    qa_input_path=paths.qa_output_path,
                    qa_v6_output_path=qa_v6_output_path,
                    show_progress=not args.no_progress,
                )

            ready_payload = {
                "dataset_name": paths.dataset_name,
                "annotation_path": paths.annotation_path,
                "qa_output_path": paths.qa_output_path,
                "qa_v6_output_path": (qa_v6_output_path if args.generate_qa_v6 else None),
                "sequence_manifest_path": paths.sequence_manifest_path,
                "sequence_manifest_dir": global_sequence_manifest_dir,
                "num_machines": num_machines,
                "qa_scan_summary": qa_scan_summary,
                "qa_v6_summary": qa_v6_summary,
                "use_gt_pose": bool(args.use_gt_pose),
                "separate_gt_pose_output": bool(args.separate_gt_pose_output),
                "created_at": time.time(),
            }
            _write_distributed_ready_marker(ready_marker_path, ready_payload)
    else:
        log_info(
            f"waiting for distributed manifest from machine-0: {ready_marker_path}"
        )
        ready_payload = _wait_for_distributed_ready_marker(
            marker_path=ready_marker_path,
            timeout_sec=args.manifest_wait_timeout_sec,
            poll_sec=args.manifest_poll_sec,
        )

    qa_scan_summary = ready_payload["qa_scan_summary"]
    if qa_scan_summary["unique_sequences"] <= 0:
        raise RuntimeError("no valid image sequences found in QA annotation")

    machine_manifest_path = _worker_sequence_manifest_path(
        global_sequence_manifest_dir,
        machine_rank,
    )
    local_shard_counts = _split_machine_manifest_to_local_workers(
        machine_manifest_path=machine_manifest_path,
        local_sequence_manifest_dir=local_sequence_manifest_dir,
        num_local_workers=len(worker_specs),
    )

    active_specs: list[tuple[int, int]] = []
    for worker_rank, gpu_id in worker_specs:
        assigned_records = int(local_shard_counts[worker_rank])
        if assigned_records <= 0:
            continue
        active_specs.append((worker_rank, gpu_id))
        log_info(
            f"machine-{machine_rank} worker-{worker_rank} -> cuda:{gpu_id}, "
            f"assigned_records={assigned_records}"
        )

    local_total_records = int(sum(local_shard_counts))

    processed_hw = _infer_processed_hw(args.input_mode, args.target_size)
    encoder_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = [encoder_hw[0] // 14, encoder_hw[1] // 14]
    run_config = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "camera_image_root": resolved_camera_image_root,
        "qa_output_path": paths.qa_output_path,
        "qa_v6_output_path": ready_payload.get("qa_v6_output_path"),
        "sequence_manifest_path": paths.sequence_manifest_path,
        "global_sequence_manifest_dir": global_sequence_manifest_dir,
        "machine_manifest_path": machine_manifest_path,
        "local_sequence_manifest_dir": local_sequence_manifest_dir,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "workers_per_gpu": int(args.workers_per_gpu),
        "num_workers": len(active_specs),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "requested_cam_token_mode": args.cam_token_mode,
        "effective_cam_token_mode": effective_cam_token_mode,
        "use_gt_pose": bool(args.use_gt_pose),
        "separate_gt_pose_output": bool(args.separate_gt_pose_output),
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "resume_behavior": "skip valid existing npz, regenerate invalid existing npz",
        "reuse_manifest": bool(args.reuse_manifest),
        "qa_scan": qa_scan_summary,
        "local_shard_counts": local_shard_counts,
        "local_total_records": local_total_records,
        "shape_hint": {
            "processed_image_hw": list(processed_hw),
            "encoder_input_hw": list(encoder_hw),
            "patch_grid_hw": list(patch_grid_hw),
            "embedding_shape_per_sample": [1, "num_views", int(patch_grid_hw[0] * patch_grid_hw[1]), 768],
        },
        "worker_assignment": [
            {
                "worker_rank": worker_rank,
                "gpu_id": gpu_id,
                "assigned_records": int(local_shard_counts[worker_rank]),
            }
            for worker_rank, gpu_id in active_specs
        ],
    }
    write_extract_config(_machine_config_path(paths, num_machines, machine_rank), run_config)
    if num_machines == 1:
        write_extract_config(paths.config_path, run_config)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes: list[mp.Process] = []
    started_at = time.time()
    for worker_rank, gpu_id in active_specs:
        proc = ctx.Process(
            target=_worker_main,
            args=(
                machine_rank,
                num_machines,
                worker_rank,
                gpu_id,
                paths,
                local_sequence_manifest_dir,
                resolved_camera_image_root,
                args.checkpoint_path,
                args.attention_type,
                args.input_mode,
                args.target_size,
                effective_cam_token_mode,
                args.use_gt_pose,
                args.storage_dtype,
                args.overwrite,
                args.startup_delay_sec,
                progress_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    live_counts = {
        "processed": 0,
        "failed": 0,
        "skipped_existing": 0,
        "regenerate_invalid_existing": 0,
    }
    worker_statuses: dict[str, str] = {}
    total_records = local_total_records
    overall_bar = tqdm(
        total=total_records,
        desc=f"{paths.dataset_name} m{machine_rank}/{num_machines} extract",
        unit="sample",
        **tqdm_options(disable=(args.no_progress or total_records == 0)),
    )
    overall_bar.set_postfix(
        processed=0,
        failed=0,
        skipped=0,
        invalid=0,
        workers=f"0/{len(active_specs)}",
        refresh=True,
    )
    overall_bar.refresh()
    last_bar_refresh_at = time.time()
    last_wait_log_at = 0.0
    worker_done_count = 0
    while worker_done_count < len(active_specs):
        try:
            event = progress_queue.get(timeout=1.0)
        except Empty:
            now = time.time()
            if total_records > 0 and now - last_bar_refresh_at >= 0.5:
                overall_bar.refresh()
                last_bar_refresh_at = now
            if len(active_specs) > 0 and now - last_wait_log_at >= 20.0:
                shard_ready = sum(1 for status in worker_statuses.values() if status in {"shard_ready", "loading_model", "model_ready", "done"})
                ready_workers = sum(1 for status in worker_statuses.values() if status in {"model_ready", "done"})
                log_info(
                    f"waiting for worker progress: shard_ready={shard_ready}/{len(active_specs)}, "
                    f"model_ready={ready_workers}/{len(active_specs)}, finished_workers={worker_done_count}"
                )
                last_wait_log_at = now
            continue

        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        if kind == "worker_status":
            label = str(event.get("label", "worker"))
            status = str(event.get("status", "unknown"))
            worker_statuses[label] = status
            if status == "loading_shard":
                log_info(f"{label} is loading shard manifest on {event.get('device', 'unknown')}")
            elif status == "shard_ready":
                log_info(
                    f"{label} shard ready on {event.get('device', 'unknown')}, "
                    f"assigned_records={event.get('assigned_records', 'unknown')}"
                )
            elif status == "loading_model":
                log_info(
                    f"{label} is loading model on {event.get('device', 'unknown')} "
                    f"(assigned_records={event.get('total_records', 'unknown')})"
                )
            elif status == "model_ready":
                suffix = ""
                if event.get("model_load_sec") is not None:
                    suffix = f", model_load_sec={event.get('model_load_sec')}"
                log_info(f"{label} model ready on {event.get('device', 'unknown')}{suffix}")

            done = live_counts["processed"] + live_counts["failed"] + live_counts["skipped_existing"]
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                skipped=live_counts["skipped_existing"],
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status in {'model_ready', 'done'})}/{len(active_specs)}",
                refresh=True,
            )
            overall_bar.n = done
            overall_bar.refresh()
            last_bar_refresh_at = time.time()
            continue

        if kind == "sample":
            status = str(event.get("status", ""))
            if status in live_counts:
                live_counts[status] += 1
            if status in {"processed", "failed", "skipped_existing"}:
                overall_bar.update(1)
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                skipped=live_counts["skipped_existing"],
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status in {'model_ready', 'done'})}/{len(active_specs)}",
                refresh=True,
            )
            last_bar_refresh_at = time.time()
            continue

        if kind == "worker_done":
            worker_done_count += 1
            label = str(event.get("label", "worker"))
            worker_statuses[label] = "done"
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                skipped=live_counts["skipped_existing"],
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status in {'model_ready', 'done'})}/{len(active_specs)}",
                refresh=True,
            )
            last_bar_refresh_at = time.time()

    overall_bar.close()

    exitcodes: list[int] = [0 for _ in worker_specs]
    for proc, (worker_rank, _gpu_id) in zip(processes, active_specs):
        proc.join()
        exitcodes[worker_rank] = int(proc.exitcode)

    worker_summary = _aggregate_worker_summaries(
        paths=paths,
        num_machines=num_machines,
        machine_rank=machine_rank,
        worker_specs=active_specs,
        process_exitcodes=exitcodes,
    )
    summary = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "qa_v6_output_path": ready_payload.get("qa_v6_output_path"),
        "sequence_manifest_path": paths.sequence_manifest_path,
        "global_sequence_manifest_dir": global_sequence_manifest_dir,
        "machine_manifest_path": machine_manifest_path,
        "local_sequence_manifest_dir": local_sequence_manifest_dir,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "workers_per_gpu": int(args.workers_per_gpu),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "requested_cam_token_mode": args.cam_token_mode,
        "effective_cam_token_mode": effective_cam_token_mode,
        "use_gt_pose": bool(args.use_gt_pose),
        "separate_gt_pose_output": bool(args.separate_gt_pose_output),
        "camera_image_root": resolved_camera_image_root,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "reuse_manifest": bool(args.reuse_manifest),
        "qa_scan": qa_scan_summary,
        "local_shard_counts": local_shard_counts,
        "local_total_records": local_total_records,
        "ready_marker_path": ready_marker_path,
        "shape_hint": run_config["shape_hint"],
        "elapsed_sec": round(time.time() - started_at, 3),
        **worker_summary,
    }
    machine_summary_path = _machine_summary_path(paths, num_machines, machine_rank)
    write_summary(machine_summary_path, summary)
    if num_machines == 1:
        write_summary(paths.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
