from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lagernvs_feature_utils import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LATENT_ROOT,
    build_qa_latent_manifest,
    compute_encoder_resize_hw,
    configure_stdout_for_tqdm,
    ensure_output_dirs,
    load_sample_records_from_sequence_manifest,
    log_info,
    process_records,
    resolve_qa_latent_paths,
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
    return int(DEFAULT_CAMERA["height"]), int(DEFAULT_CAMERA["width"])


def main() -> None:
    configure_stdout_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Extract LagerNVS embeddings from unique image sequences in QA_v4_w_thinking.json"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PAP_512x384_fov_90",
        help="数据集名称；会从 qwenvl.data 解析 annotation_path 和 image_root",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="可选；直接指定 QA_v4_w_thinking.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="可选；覆盖图片根目录",
    )
    parser.add_argument(
        "--qa_output_path",
        type=str,
        default=None,
        help="可选；覆盖 QA_v5_w_latent.json 输出路径",
    )
    parser.add_argument(
        "--latent_root",
        type=str,
        default=str(DEFAULT_LATENT_ROOT),
        help="latent_data 根目录",
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
        help="默认 zeros，表示相机内外参未知",
    )
    parser.add_argument(
        "--storage_dtype",
        type=str,
        default="float16",
        choices=STORAGE_DTYPE_CHOICES,
        help="embedding 存储精度",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多扫描多少条 QA，便于 debug",
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

    paths = resolve_qa_latent_paths(
        dataset_name=args.dataset_name,
        latent_root=args.latent_root,
        annotation_path=args.annotation_path,
        image_root=args.image_root,
        qa_output_path=args.qa_output_path,
    )
    ensure_output_dirs(paths)

    log_info(f"loading QA annotation from {paths.annotation_path}")
    qa_scan_summary = build_qa_latent_manifest(
        qa_input_path=paths.annotation_path,
        qa_output_path=paths.qa_output_path,
        sequence_manifest_path=paths.sequence_manifest_path,
        sequence_manifest_dir=paths.sequence_manifest_dir,
        num_sequence_shards=1,
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

    shard_path = Path(paths.sequence_manifest_dir) / "worker_00.jsonl"
    records = load_sample_records_from_sequence_manifest(
        manifest_path=shard_path,
        image_root=paths.image_root,
        latent_dir=paths.latent_dir,
        default_camera=DEFAULT_CAMERA,
    )
    log_info(f"sequence manifest loaded: unique_records={len(records)}")

    processed_hw = _infer_processed_hw(args.input_mode, args.target_size, records)
    encoder_hw = compute_encoder_resize_hw(processed_hw)
    patch_grid_hw = [encoder_hw[0] // 14, encoder_hw[1] // 14]
    progress_path = str(Path(paths.progress_dir) / "single_process_qa.progress.jsonl")
    run_config = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "sequence_manifest_path": paths.sequence_manifest_path,
        "latent_dir": paths.latent_dir,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "progress_path": progress_path,
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
    }
    write_extract_config(paths.config_path, run_config)

    summary = process_records(
        records=records,
        checkpoint_path=args.checkpoint_path,
        device_str=args.device,
        attention_type=args.attention_type,
        input_mode=args.input_mode,
        target_size=args.target_size,
        cam_token_mode=args.cam_token_mode,
        storage_dtype=args.storage_dtype,
        overwrite=args.overwrite,
        error_path=str(Path(paths.error_dir) / "single_process_qa.errors.jsonl"),
        progress_path=progress_path,
        show_progress=not args.no_progress,
        progress_desc=f"{paths.dataset_name} extract",
        log_every=100,
    )
    final_summary = {
        "dataset_name": paths.dataset_name,
        "annotation_path": paths.annotation_path,
        "image_root": paths.image_root,
        "qa_output_path": paths.qa_output_path,
        "sequence_manifest_path": paths.sequence_manifest_path,
        "latent_dir": paths.latent_dir,
        "shape_hint": run_config["shape_hint"],
        "qa_scan": qa_scan_summary,
        **summary,
    }
    write_summary(paths.summary_path, final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
