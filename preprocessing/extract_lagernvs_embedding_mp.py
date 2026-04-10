"""
/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_mp.py \
  --dataset_name PAP_512x384_fov_90 \
  --gpus 0,1,2,3 \
  --workers_per_gpu 8

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_mp.py \
  --dataset_name PAP_512x384_fov_90 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 8

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_mp.py \
  --dataset_name PAP_retrieval_512x384_fov_90 \
  --gpus 0,1,2,3 \
  --workers_per_gpu 6


/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_mp.py \
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
    DEFAULT_CAMERA_DATA_ROOT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LATENT_ROOT,
    DatasetPaths,
    SampleRecord,
    build_sample_records_multiprocess,
    compute_encoder_resize_hw,
    configure_stdout_for_tqdm,
    ensure_output_dirs,
    format_count_summary,
    load_metadata,
    log_info,
    process_records,
    resolve_dataset_paths,
    scan_records_multiprocess,
    shard_records,
    slice_metadata,
    write_extract_config,
    write_manifest,
    write_summary,
    tqdm_options,
)


CAM_TOKEN_CHOICES = (
    "zeros",
    "zeros_world_scale",
    "metadata_pose_world_scale",
    "metadata_pose_zero_scale",
)
STORAGE_DTYPE_CHOICES = ("float16", "int8", "float32")


def _parse_gpu_ids(gpus: str) -> list[int]:
    text = str(gpus or "").strip()
    if not text:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _infer_processed_hw(input_mode: str, target_size: int, records) -> tuple[int, int]:
    if input_mode == "square_crop":
        return int(target_size), int(target_size)
    if records:
        camera = records[0].camera
        width = int(round(float(camera.get("width", target_size))))
        height = int(round(float(camera.get("height", target_size))))
        if width >= height:
            new_width = int(target_size)
            new_height = int(round(height * (target_size / width) / 8) * 8)
        else:
            new_height = int(target_size)
            new_width = int(round(width * (target_size / height) / 8) * 8)
        return new_height, new_width
    return int(target_size), int(target_size)


def _worker_summary_path(paths: DatasetPaths, worker_rank: int) -> str:
    return str(Path(paths.latent_dir) / "_workers" / f"worker_{worker_rank:02d}.summary.json")


def _worker_error_path(paths: DatasetPaths, worker_rank: int) -> str:
    return str(Path(paths.error_dir) / f"worker_{worker_rank:02d}.errors.jsonl")


def _worker_progress_path(paths: DatasetPaths, worker_rank: int) -> str:
    return str(Path(paths.progress_dir) / f"worker_{worker_rank:02d}.progress.jsonl")


def _worker_main(
    worker_rank: int,
    gpu_id: int,
    records: list[SampleRecord],
    paths: DatasetPaths,
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
        progress_desc=f"worker-{worker_rank}",
        progress_queue=progress_queue,
        progress_queue_label=f"worker-{worker_rank}",
        log_every=0,
    )
    summary.update(
        {
            "worker_rank": int(worker_rank),
            "gpu_id": int(gpu_id),
            "assigned_records": len(records),
        }
    )
    write_summary(_worker_summary_path(paths, worker_rank), summary)


def _aggregate_worker_summaries(
    paths: DatasetPaths,
    worker_specs: list[tuple[int, int]],
    process_exitcodes: list[int],
) -> dict[str, object]:
    workers: list[dict[str, object]] = []
    totals = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "total_records": 0,
    }
    for worker_rank, gpu_id in worker_specs:
        summary_path = Path(_worker_summary_path(paths, worker_rank))
        worker_summary: dict[str, object]
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
                "missing_summary": True,
            }
        worker_summary["exitcode"] = int(process_exitcodes[worker_rank])
        workers.append(worker_summary)

        totals["processed"] += int(worker_summary.get("processed", 0))
        totals["skipped_existing"] += int(worker_summary.get("skipped_existing", 0))
        totals["failed"] += int(worker_summary.get("failed", 0))
        totals["total_records"] += int(worker_summary.get("total_records", 0))

    return {
        **totals,
        "workers": workers,
        "num_workers": len(worker_specs),
    }


def main() -> None:
    configure_stdout_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Extract LagerNVS Reconstructor embeddings with multiple GPUs/processes"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PAP_512x384_fov_90",
        help="camera_data 子目录名，例如 PAP_512x384_fov_90 或 PAP_retrieval_512x384_fov_90",
    )
    parser.add_argument(
        "--camera_data_root",
        type=str,
        default=str(DEFAULT_CAMERA_DATA_ROOT),
        help="camera_data 根目录",
    )
    parser.add_argument(
        "--latent_root",
        type=str,
        default=str(DEFAULT_LATENT_ROOT),
        help="latent_data 根目录",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="可选；直接指定 metadata.json",
    )
    parser.add_argument(
        "--camera_image_root",
        type=str,
        default=None,
        help="可选；直接指定 camera_image 根目录",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="LagerNVS checkpoint 路径",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="使用哪些 GPU，例如 0,1,2,3",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="每张卡加载几个模型/起几个 worker",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=("bidirectional_cross_attention", "full_attention"),
        help="General 512 模型使用 bidirectional_cross_attention",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="resize",
        choices=("resize", "square_crop"),
        help="图像预处理模式；当前 512x384 数据建议使用 resize",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="输入到 load_and_preprocess_images 的 target_size",
    )
    parser.add_argument(
        "--cam_token_mode",
        type=str,
        default="zeros",
        choices=CAM_TOKEN_CHOICES,
        help="默认 zeros，表示相机内外参未知；metadata_pose_* 会用 metadata 里的 yaw/pitch+intrinsics 构造 11 维 camera token",
    )
    parser.add_argument(
        "--storage_dtype",
        type=str,
        default="float16",
        choices=STORAGE_DTYPE_CHOICES,
        help="embedding 存储精度，推荐 float16；int8 会额外保存 embedding_scale",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="从 metadata 的哪个 index 开始",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="metadata 的结束 index（开区间）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理多少条，便于 debug",
    )
    parser.add_argument(
        "--scan_workers",
        type=int,
        default=os.cpu_count() or 8,
        help="预扫描 metadata/output 的进程数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 npz",
    )
    parser.add_argument(
        "--startup_delay_sec",
        type=float,
        default=3.0,
        help="worker 启动间隔，避免同时加载 4GB checkpoint 打爆 IO",
    )
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("no GPU selected")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multiprocessing extractor")

    paths = resolve_dataset_paths(
        dataset_name=args.dataset_name,
        camera_data_root=args.camera_data_root,
        latent_root=args.latent_root,
        metadata_path=args.metadata_path,
        camera_image_root=args.camera_image_root,
    )
    ensure_output_dirs(paths)
    Path(paths.latent_dir, "_workers").mkdir(parents=True, exist_ok=True)

    log_info(f"loading metadata from {paths.metadata_path}")
    metadata = load_metadata(paths.metadata_path)
    metadata_slice, start, end = slice_metadata(
        metadata,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_samples=args.max_samples,
    )
    log_info(
        f"metadata loaded: total={len(metadata)}, selected_range=[{start}, {end}), "
        f"selected_items={len(metadata_slice)}"
    )
    records = build_sample_records_multiprocess(
        metadata=metadata_slice,
        camera_image_root=paths.camera_image_root,
        latent_dir=paths.latent_dir,
        metadata_offset=start,
        num_workers=args.scan_workers,
        show_progress=True,
        progress_desc=f"{paths.dataset_name} build-records",
    )
    if not records:
        raise RuntimeError("no valid records found for extraction")
    log_info(f"record build finished: selected_records={len(records)}")

    write_manifest(
        manifest_path=paths.manifest_path,
        dataset_name=paths.dataset_name,
        metadata_path=paths.metadata_path,
        camera_image_root=paths.camera_image_root,
        records=records,
    )
    pending_records, pre_scan_summary = scan_records_multiprocess(
        records=records,
        overwrite=args.overwrite,
        num_workers=args.scan_workers,
        show_progress=True,
        progress_desc=f"{paths.dataset_name} pre-scan",
    )
    log_info(f"pre-scan summary: {format_count_summary(pre_scan_summary)}")

    processed_hw = _infer_processed_hw(args.input_mode, args.target_size, records)
    encoder_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = [encoder_hw[0] // 14, encoder_hw[1] // 14]

    worker_specs: list[tuple[int, int]] = []
    for gpu_id in gpu_ids:
        for _ in range(max(1, int(args.workers_per_gpu))):
            worker_specs.append((len(worker_specs), int(gpu_id)))

    shards = shard_records(pending_records, len(worker_specs))
    active_specs: list[tuple[int, int]] = []
    active_shards: list[list[SampleRecord]] = []
    for spec, shard in zip(worker_specs, shards):
        if shard:
            active_specs.append(spec)
            active_shards.append(shard)
    log_info(
        f"worker assignment ready: workers={len(active_specs)}, "
        f"pending_records={len(pending_records)}"
    )
    for (worker_rank, gpu_id), shard in zip(active_specs, active_shards):
        log_info(f"worker-{worker_rank} -> cuda:{gpu_id}, assigned_records={len(shard)}")

    run_config = {
        "dataset_name": paths.dataset_name,
        "metadata_path": paths.metadata_path,
        "camera_image_root": paths.camera_image_root,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "progress_dir": paths.progress_dir,
        "workers_per_gpu": int(args.workers_per_gpu),
        "num_workers": len(active_specs),
        "scan_workers": int(args.scan_workers),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "cam_token_mode": args.cam_token_mode,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "resume_behavior": "skip valid existing npz, regenerate invalid existing npz",
        "start_idx": int(start),
        "end_idx": int(end),
        "selected_records": len(records),
        "pending_records": len(pending_records),
        "pre_scan": pre_scan_summary,
        "shape_hint": {
            "processed_image_hw": list(processed_hw),
            "encoder_input_hw": list(encoder_hw),
            "patch_grid_hw": list(patch_grid_hw),
            "embedding_shape_per_sample": [
                1,
                "num_views",
                int(patch_grid_hw[0] * patch_grid_hw[1]),
                768,
            ],
        },
        "worker_assignment": [
            {
                "worker_rank": worker_rank,
                "gpu_id": gpu_id,
                "assigned_records": len(shard),
            }
            for (worker_rank, gpu_id), shard in zip(active_specs, active_shards)
        ],
    }
    write_extract_config(paths.config_path, run_config)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes: list[mp.Process] = []
    started_at = time.time()
    for (worker_rank, gpu_id), shard in zip(active_specs, active_shards):
        proc = ctx.Process(
            target=_worker_main,
            args=(
                worker_rank,
                gpu_id,
                shard,
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
    total_pending = len(pending_records)
    overall_bar = tqdm(
        total=total_pending,
        desc=f"{paths.dataset_name} extract",
        unit="sample",
        **tqdm_options(disable=(total_pending == 0)),
    )
    overall_bar.set_postfix(
        processed=0,
        failed=0,
        remaining=total_pending,
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
            if total_pending > 0 and now - last_bar_refresh_at >= 0.5:
                overall_bar.refresh()
                last_bar_refresh_at = now
            if len(active_specs) > 0 and worker_done_count < len(active_specs) and now - last_wait_log_at >= 20.0:
                loading_workers = sum(1 for status in worker_statuses.values() if status == "loading_model")
                ready_workers = sum(1 for status in worker_statuses.values() if status == "model_ready")
                log_info(
                    f"waiting for worker progress: ready_workers={ready_workers}/{len(active_specs)}, "
                    f"loading_workers={loading_workers}, finished_workers={worker_done_count}"
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
            model_load_sec = event.get("model_load_sec")
            if status == "loading_model":
                log_info(
                    f"{label} is loading model on {event.get('device', 'unknown')} "
                    f"(assigned_records={event.get('total_records', 'unknown')})"
                )
            elif status == "model_ready":
                load_suffix = ""
                if model_load_sec is not None:
                    load_suffix = f", model_load_sec={model_load_sec}"
                log_info(
                    f"{label} model ready on {event.get('device', 'unknown')}{load_suffix}"
                )
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                remaining=max(
                    0,
                    total_pending
                    - (
                        live_counts["processed"]
                        + live_counts["failed"]
                        + live_counts["skipped_existing"]
                    ),
                ),
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status == 'model_ready')}/{len(active_specs)}",
                refresh=True,
            )
            last_bar_refresh_at = time.time()
            continue
        if kind == "sample":
            status = str(event.get("status", ""))
            if status in live_counts:
                live_counts[status] += 1
            if status in {"processed", "failed", "skipped_existing"}:
                overall_bar.update(1)
            done = (
                live_counts["processed"]
                + live_counts["failed"]
                + live_counts["skipped_existing"]
            )
            remaining = max(0, total_pending - done)
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                remaining=remaining,
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status == 'model_ready')}/{len(active_specs)}",
                refresh=True,
            )
            last_bar_refresh_at = time.time()
            if done > 0 and done % 200 == 0:
                log_info(
                    f"live progress: processed={live_counts['processed']}, "
                    f"failed={live_counts['failed']}, remaining={remaining}, "
                    f"invalid_rebuilt={live_counts['regenerate_invalid_existing']}"
                )
        elif kind == "worker_done":
            worker_done_count += 1
            label = str(event.get("label", "worker"))
            worker_statuses[label] = "done"
            overall_bar.set_postfix(
                processed=live_counts["processed"],
                failed=live_counts["failed"],
                remaining=max(
                    0,
                    total_pending
                    - (
                        live_counts["processed"]
                        + live_counts["failed"]
                        + live_counts["skipped_existing"]
                    ),
                ),
                invalid=live_counts["regenerate_invalid_existing"],
                workers=f"{sum(1 for status in worker_statuses.values() if status in {'model_ready', 'done'})}/{len(active_specs)}",
                refresh=True,
            )
            last_bar_refresh_at = time.time()

    overall_bar.close()

    exitcodes: list[int] = [0 for _ in active_specs]
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
        "metadata_path": paths.metadata_path,
        "camera_image_root": paths.camera_image_root,
        "latent_dir": paths.latent_dir,
        "manifest_path": paths.manifest_path,
        "config_path": paths.config_path,
        "checkpoint_path": args.checkpoint_path,
        "gpus": gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "scan_workers": int(args.scan_workers),
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "cam_token_mode": args.cam_token_mode,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "start_idx": int(start),
        "end_idx": int(end),
        "selected_records": len(records),
        "pending_records": len(pending_records),
        "pre_scan": pre_scan_summary,
        "shape_hint": run_config["shape_hint"],
        "elapsed_sec": round(time.time() - started_at, 3),
        **worker_summary,
    }
    write_summary(paths.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
