"""
/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp.py \
  --dataset_name hstar_sft_512x384_fov_90 \
  --gpus 0,1,2,3 \
  --workers_per_gpu 8

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp.py \
  --dataset_name PAP_512x384_fov_90 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 8

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp.py \
  --dataset_name raw_pano_512x384_fov_90 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 8

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp.py \
  --dataset_name PAP_retrieval_512x384_fov_90 \
  --gpus 0,1,2,3 \
  --workers_per_gpu 6


/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_qa_mp.py \
  --dataset_name PAP_retrieval_512x384_fov_90 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 6

"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty
from typing import Any

import torch
from tqdm import tqdm

from lagernvs_feature_utils import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LATENT_ROOT,
    QALatentPaths,
    build_qa_latent_manifest,
    compute_encoder_resize_hw,
    configure_stdout_for_tqdm,
    ensure_output_dirs,
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


def _worker_summary_path(paths: QALatentPaths, worker_rank: int) -> str:
    return str(Path(paths.latent_dir) / "_workers" / f"qa_worker_{worker_rank:02d}.summary.json")


def _worker_error_path(paths: QALatentPaths, worker_rank: int) -> str:
    return str(Path(paths.error_dir) / f"qa_worker_{worker_rank:02d}.errors.jsonl")


def _worker_progress_path(paths: QALatentPaths, worker_rank: int) -> str:
    return str(Path(paths.progress_dir) / f"qa_worker_{worker_rank:02d}.progress.jsonl")


def _worker_sequence_manifest_path(paths: QALatentPaths, worker_rank: int) -> str:
    return str(Path(paths.sequence_manifest_dir) / f"worker_{worker_rank:02d}.jsonl")


def _worker_main(
    worker_rank: int,
    gpu_id: int,
    paths: QALatentPaths,
    checkpoint_path: str,
    attention_type: str,
    input_mode: str,
    target_size: int,
    cam_token_mode: str,
    storage_dtype: str,
    overwrite: bool,
    startup_delay_sec: float,
    progress_queue: Any = None,
) -> None:
    configure_stdout_for_tqdm()
    Path(paths.latent_dir, "_workers").mkdir(parents=True, exist_ok=True)
    if startup_delay_sec > 0:
        time.sleep(float(startup_delay_sec) * float(worker_rank))

    manifest_path = _worker_sequence_manifest_path(paths, worker_rank)
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
        error_path=_worker_error_path(paths, worker_rank),
        progress_path=_worker_progress_path(paths, worker_rank),
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
    write_summary(_worker_summary_path(paths, worker_rank), summary)


def _aggregate_worker_summaries(
    paths: QALatentPaths,
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
        summary_path = Path(_worker_summary_path(paths, worker_rank))
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
    parser.add_argument("--qa_output_path", type=str, default=None, help="可选；覆盖 QA_v5_w_latent.json 输出路径")
    parser.add_argument("--latent_root", type=str, default=str(DEFAULT_LATENT_ROOT), help="latent_data 根目录")
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="checkpoint 路径")
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
    parser.add_argument("--storage_dtype", type=str, default="float16", choices=STORAGE_DTYPE_CHOICES)
    parser.add_argument("--max_samples", type=int, default=None, help="最多扫描多少条 QA，便于 debug")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 npz")
    parser.add_argument(
        "--startup_delay_sec",
        type=float,
        default=3.0,
        help="worker 启动间隔，避免同时加载 checkpoint 打爆 IO",
    )
    parser.add_argument("--no_progress", action="store_true", help="关闭 tqdm 进度条")
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("no GPU selected")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multiprocessing extractor")

    worker_specs: list[tuple[int, int]] = []
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
    ensure_output_dirs(paths)
    Path(paths.latent_dir, "_workers").mkdir(parents=True, exist_ok=True)

    log_info(f"loading QA annotation from {paths.annotation_path}")
    qa_scan_summary = build_qa_latent_manifest(
        qa_input_path=paths.annotation_path,
        qa_output_path=paths.qa_output_path,
        sequence_manifest_path=paths.sequence_manifest_path,
        sequence_manifest_dir=paths.sequence_manifest_dir,
        num_sequence_shards=len(worker_specs),
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

    active_specs: list[tuple[int, int]] = []
    for worker_rank, gpu_id in worker_specs:
        assigned_records = int(qa_scan_summary["shard_counts"][worker_rank])
        if assigned_records <= 0:
            continue
        active_specs.append((worker_rank, gpu_id))
        log_info(f"worker-{worker_rank} -> cuda:{gpu_id}, assigned_records={assigned_records}")

    processed_hw = _infer_processed_hw(args.input_mode, args.target_size)
    encoder_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = [encoder_hw[0] // 14, encoder_hw[1] // 14]
    run_config = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "sequence_manifest_path": paths.sequence_manifest_path,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "num_workers": len(active_specs),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "cam_token_mode": args.cam_token_mode,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "resume_behavior": "skip valid existing npz, regenerate invalid existing npz",
        "qa_scan": qa_scan_summary,
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
                "assigned_records": int(qa_scan_summary["shard_counts"][worker_rank]),
            }
            for worker_rank, gpu_id in active_specs
        ],
    }
    write_extract_config(paths.config_path, run_config)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes: list[mp.Process] = []
    started_at = time.time()
    for worker_rank, gpu_id in active_specs:
        proc = ctx.Process(
            target=_worker_main,
            args=(
                worker_rank,
                gpu_id,
                paths,
                args.checkpoint_path,
                args.attention_type,
                args.input_mode,
                args.target_size,
                args.cam_token_mode,
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
    total_records = int(qa_scan_summary["unique_sequences"])
    overall_bar = tqdm(
        total=total_records,
        desc=f"{paths.dataset_name} extract",
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
        worker_specs=active_specs,
        process_exitcodes=exitcodes,
    )
    summary = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "sequence_manifest_path": paths.sequence_manifest_path,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "cam_token_mode": args.cam_token_mode,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "qa_scan": qa_scan_summary,
        "shape_hint": run_config["shape_hint"],
        "elapsed_sec": round(time.time() - started_at, 3),
        **worker_summary,
    }
    write_summary(paths.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
