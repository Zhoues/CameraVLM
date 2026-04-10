from __future__ import annotations

import argparse
import json
import os

import torch

from lagernvs_feature_utils import (
    DEFAULT_CAMERA_DATA_ROOT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LATENT_ROOT,
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
    slice_metadata,
    write_extract_config,
    write_manifest,
    write_summary,
)


CAM_TOKEN_CHOICES = (
    "zeros",
    "zeros_world_scale",
    "metadata_pose_world_scale",
    "metadata_pose_zero_scale",
)
STORAGE_DTYPE_CHOICES = ("float16", "int8", "float32")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


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


def main() -> None:
    configure_stdout_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Extract LagerNVS Reconstructor embeddings for one camera_data dataset"
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
        "--device",
        type=str,
        default=_default_device(),
        help="推理设备，例如 cuda:0",
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
        "--no_progress",
        action="store_true",
        help="关闭 tqdm 进度条",
    )
    args = parser.parse_args()

    paths = resolve_dataset_paths(
        dataset_name=args.dataset_name,
        camera_data_root=args.camera_data_root,
        latent_root=args.latent_root,
        metadata_path=args.metadata_path,
        camera_image_root=args.camera_image_root,
    )
    ensure_output_dirs(paths)

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
        show_progress=not args.no_progress,
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
        show_progress=not args.no_progress,
        progress_desc=f"{paths.dataset_name} pre-scan",
    )
    log_info(f"pre-scan summary: {format_count_summary(pre_scan_summary)}")

    processed_hw = _infer_processed_hw(args.input_mode, args.target_size, records)
    encoder_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = [encoder_hw[0] // 14, encoder_hw[1] // 14]

    run_config = {
        "dataset_name": paths.dataset_name,
        "metadata_path": paths.metadata_path,
        "camera_image_root": paths.camera_image_root,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "progress_path": f"{paths.progress_dir}/single_process.progress.jsonl",
        "attention_type": args.attention_type,
        "input_mode": args.input_mode,
        "target_size": int(args.target_size),
        "cam_token_mode": args.cam_token_mode,
        "storage_dtype": args.storage_dtype,
        "overwrite": bool(args.overwrite),
        "scan_workers": int(args.scan_workers),
        "resume_behavior": "skip valid existing npz, regenerate invalid existing npz",
        "start_idx": int(start),
        "end_idx": int(end),
        "selected_records": len(records),
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
    }
    write_extract_config(paths.config_path, run_config)

    error_path = f"{paths.error_dir}/single_process.errors.jsonl"
    progress_path = run_config["progress_path"]
    log_info(
        f"start extracting on {args.device}: pending_records={len(pending_records)}, "
        f"selected_records={len(records)}, error_path={error_path}"
    )
    summary = process_records(
        records=pending_records,
        checkpoint_path=args.checkpoint_path,
        device_str=args.device,
        attention_type=args.attention_type,
        input_mode=args.input_mode,
        target_size=args.target_size,
        cam_token_mode=args.cam_token_mode,
        storage_dtype=args.storage_dtype,
        overwrite=args.overwrite,
        error_path=error_path,
        progress_path=progress_path,
        show_progress=not args.no_progress,
        progress_desc=f"{paths.dataset_name} embeddings",
        log_every=50,
    )
    summary.update(
        {
            "dataset_name": paths.dataset_name,
            "metadata_path": paths.metadata_path,
            "camera_image_root": paths.camera_image_root,
            "latent_dir": paths.latent_dir,
            "manifest_path": paths.manifest_path,
            "config_path": paths.config_path,
            "error_path": error_path,
            "progress_path": progress_path,
            "scan_workers": int(args.scan_workers),
            "selected_records": len(records),
            "pending_records": len(pending_records),
            "pre_scan": pre_scan_summary,
            "shape_hint": run_config["shape_hint"],
        }
    )
    write_summary(paths.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
