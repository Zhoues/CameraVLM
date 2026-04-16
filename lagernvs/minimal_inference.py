# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal end-to-end inference script for LagerNVS.

This script demonstrates the full pipeline:
  1. Load input images
  2. Create a target camera trajectory (using VGGT for pose estimation)
  3. Load the LagerNVS checkpoint
  4. Render novel views
  5. Save output as an MP4 video

Prerequisites:
  - GPU with CUDA support (bfloat16 on Ampere+ GPUs, float16 otherwise)
  - If you do not pass local checkpoints, a HuggingFace token with access to
    the gated model repo may still be required.

Usage:
  python minimal_inference.py --images path/to/img1.png path/to/img2.png
  python minimal_inference.py --images images/input_000000.png images/input_000001.png
  python minimal_inference.py --images a.png b.png --model_ckpt /path/to/model.pt --vggt_ckpt /path/to/vggt.pt

Available checkpoints (set via --model_repo and --attention_type):
  General (512px):  facebook/lagernvs_general_512   attention=bidirectional_cross_attention
  Re10k (256px):    facebook/lagernvs_re10k_2v_256  attention=full_attention
  DL3DV (256px):    facebook/lagernvs_dl3dv_2-6_v_256 attention=bidirectional_cross_attention
"""

import argparse
import json
import math
from pathlib import Path

import torch
from eval.export import save_video
from huggingface_hub import hf_hub_download
from models.encoder_decoder import EncDec_VitB8
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vis import (
    _load_vggt_state_dict,
    compute_plucker_coordinates,
    create_target_camera_path,
    render_chunked,
)


def _rotation_x(angle_rad, device, dtype):
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        device=device,
        dtype=dtype,
    )


def _rotation_y(angle_rad, device, dtype):
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    return torch.tensor(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        device=device,
        dtype=dtype,
    )


def _build_c2w_from_yaw_pitch(yaw_deg, pitch_deg, device, dtype):
    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(pitch_deg))
    # The pano crop generator applies row-vector rotations with positive yaw
    # turning right and positive pitch turning up. Converting that to LagerNVS'
    # column-vector camera math gives this c2w rotation directly.
    rotation = _rotation_y(yaw_rad, device, dtype) @ _rotation_x(
        pitch_rad, device, dtype
    )
    extrinsic = torch.zeros(4, 4, device=device, dtype=dtype)
    extrinsic[:3, :3] = rotation
    extrinsic[3, 3] = 1.0
    return extrinsic


def _invert_rigid(w2c):
    c2w = torch.zeros_like(w2c)
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    c2w[..., :3, :3] = R_inv
    c2w[..., :3, 3:] = -R_inv @ t
    c2w[..., 3, 3] = 1.0
    return c2w


def _shortest_yaw_delta_deg(src_yaw, dst_yaw):
    return ((float(dst_yaw) - float(src_yaw) + 180.0) % 360.0) - 180.0


def _normalize_c2w_for_lagernvs(c2w, num_cond_views):
    c2w = c2w.clone()
    first_cam_inv = torch.linalg.inv(c2w[:, 0:1, :, :])
    c2w = first_cam_inv @ c2w

    if num_cond_views <= 1:
        scale_tokens = torch.tensor([0.0, 1.0], device=c2w.device, dtype=torch.float32)
        return c2w, scale_tokens

    cond_norms = torch.norm(c2w[:, :num_cond_views, :3, 3], dim=-1)
    max_norm = torch.max(cond_norms)
    if float(max_norm) <= 1e-6:
        scale_tokens = torch.tensor([0.0, 1.0], device=c2w.device, dtype=torch.float32)
        return c2w, scale_tokens

    scene_scale = torch.clamp(1.35 * max_norm, min=1e-6)
    c2w[:, :, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(c2w[:, :num_cond_views, :3, 3], dim=-1)).item()
    scale_tokens = torch.tensor(
        [camera_scale, 0.0], device=c2w.device, dtype=torch.float32
    )
    return c2w, scale_tokens


def _infer_annotation_path(image_names):
    parent_dirs = {str(Path(image_name).resolve().parent) for image_name in image_names}
    if len(parent_dirs) != 1:
        raise ValueError(
            "Could not infer a unique annotation.json path from the provided images. "
            "Please pass --annotation_json explicitly."
        )
    annotation_path = Path(next(iter(parent_dirs))) / "annotation.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"annotation.json not found: {annotation_path}")
    return annotation_path


def _load_annotation_payload(annotation_json):
    with open(annotation_json, "r", encoding="utf-8") as f_in:
        return json.load(f_in)


def _scale_intrinsics_from_annotation(annotation_camera, image_size_hw, device, dtype):
    src_w = float(annotation_camera["width"])
    src_h = float(annotation_camera["height"])
    tgt_h, tgt_w = image_size_hw
    scale_x = float(tgt_w) / src_w
    scale_y = float(tgt_h) / src_h
    return torch.tensor(
        [
            float(annotation_camera["fx"]) * scale_x,
            float(annotation_camera["fy"]) * scale_y,
            float(annotation_camera["cx"]) * scale_x,
            float(annotation_camera["cy"]) * scale_y,
        ],
        device=device,
        dtype=dtype,
    )


def _load_gt_camera_data(image_names, annotation_json, image_size_hw, device, dtype):
    annotation = _load_annotation_payload(annotation_json)
    frame_lookup = {
        Path(frame["image"]).name: tuple(frame["pose"]) for frame in annotation["frames"]
    }

    poses = []
    for image_name in image_names:
        basename = Path(image_name).name
        if basename not in frame_lookup:
            raise KeyError(f"Image {basename} is missing from {annotation_json}")
        poses.append(frame_lookup[basename])

    intrinsics_single = _scale_intrinsics_from_annotation(
        annotation["camera"], image_size_hw, device, dtype
    )
    intrinsics = intrinsics_single.unsqueeze(0).repeat(len(image_names), 1)

    c2w = torch.stack(
        [
            _build_c2w_from_yaw_pitch(yaw, pitch, device=device, dtype=dtype)
            for yaw, pitch in poses
        ],
        dim=0,
    ).unsqueeze(0)
    c2w, scale_tokens = _normalize_c2w_for_lagernvs(c2w, num_cond_views=len(image_names))
    pose_tokens = extri_intri_to_pose_encoding(
        c2w[:, :, :3, :],
        intrinsics.unsqueeze(0),
        image_size_hw=image_size_hw,
    ).float()
    return {
        "poses": poses,
        "c2w": c2w,
        "fxfycxcy": intrinsics.unsqueeze(0),
        "pose_tokens": pose_tokens,
        "scale_tokens": scale_tokens,
    }


def _build_gt_target_rays(poses, fxfycxcy, video_length, image_size_hw, device, dtype):
    if len(poses) < 2:
        raise ValueError("GT trajectory generation requires at least 2 images.")

    yaw_0, pitch_0 = poses[0]
    yaw_1, pitch_1 = poses[1]
    delta_yaw = _shortest_yaw_delta_deg(yaw_0, yaw_1)
    delta_pitch = float(pitch_1) - float(pitch_0)

    half_length = video_length // 2
    t_forward = torch.linspace(0.0, 1.0, half_length, device=device, dtype=dtype)
    t_back = torch.linspace(
        1.0, 0.0, video_length - half_length, device=device, dtype=dtype
    )
    t_all = torch.cat([t_forward, t_back], dim=0)

    target_c2w = []
    for t in t_all.tolist():
        yaw_t = float(yaw_0) + delta_yaw * t
        pitch_t = float(pitch_0) + delta_pitch * t
        target_c2w.append(
            _build_c2w_from_yaw_pitch(yaw_t, pitch_t, device=device, dtype=dtype)
        )
    target_c2w = torch.stack(target_c2w, dim=0).unsqueeze(0)
    first_cam_inv = torch.linalg.inv(target_c2w[:, 0:1, :, :])
    target_c2w = first_cam_inv @ target_c2w

    target_fxfycxcy = fxfycxcy[:, 0:1, :].expand(1, video_length, 4).contiguous()
    return compute_plucker_coordinates(target_c2w, target_fxfycxcy, image_size_hw)


def _build_vggt_condition_tokens(
    image_names,
    image_size_hw,
    device,
    dtype,
    mode,
    vggt_checkpoint_path,
):
    images_vggt = load_and_preprocess_images(
        image_names, mode=mode, target_size=518, patch_size=14
    ).to(device)

    vggt_model = VGGT(pred_cameras=True)
    vggt_model.load_state_dict(_load_vggt_state_dict(vggt_checkpoint_path), strict=False)
    vggt_model.to(device)
    vggt_model.eval()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            pose_enc = vggt_model(images_vggt)

    if pose_enc.dim() == 2:
        pose_enc = pose_enc.unsqueeze(0)

    extrinsics_w2c, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_size_hw
    )
    w2c_full = torch.zeros(
        extrinsics_w2c.shape[:2] + (4, 4),
        device=extrinsics_w2c.device,
        dtype=extrinsics_w2c.dtype,
    )
    w2c_full[:, :, :3, :] = extrinsics_w2c
    w2c_full[:, :, 3, 3] = 1.0

    c2w = _invert_rigid(w2c_full)
    num_cond_views = len(image_names)
    c2w, scale_tokens = _normalize_c2w_for_lagernvs(c2w, num_cond_views=num_cond_views)

    fxfycxcy = torch.stack(
        [
            intrinsics[:, :, 0, 0],
            intrinsics[:, :, 1, 1],
            intrinsics[:, :, 0, 2],
            intrinsics[:, :, 1, 2],
        ],
        dim=-1,
    )
    pose_tokens = extri_intri_to_pose_encoding(
        c2w[:, :, :3, :],
        fxfycxcy,
        image_size_hw=image_size_hw,
    ).float()
    scale_tokens = scale_tokens.view(1, 1, 2).expand(1, num_cond_views, 2)
    return torch.cat([pose_tokens, scale_tokens], dim=-1)


def _assemble_cam_tokens(cond_tokens, video_length):
    batch_size, num_cond_views, _ = cond_tokens.shape
    total_views = num_cond_views + video_length
    cam_tokens = torch.zeros(batch_size, total_views, 11, dtype=torch.float32)
    cam_tokens[:, :num_cond_views, :] = cond_tokens.float().cpu()
    cam_tokens[:, :, 9:] = cond_tokens[:, 0:1, 9:].float().cpu().expand(
        batch_size, total_views, 2
    )
    return cam_tokens


def _build_compare_inputs(
    image_names,
    annotation_json,
    image_size_hw,
    video_length,
    device,
    dtype,
    mode,
    vggt_checkpoint_path,
):
    gt_data = _load_gt_camera_data(
        image_names=image_names,
        annotation_json=annotation_json,
        image_size_hw=image_size_hw,
        device=device,
        dtype=torch.float32,
    )
    target_rays = _build_gt_target_rays(
        poses=gt_data["poses"],
        fxfycxcy=gt_data["fxfycxcy"],
        video_length=video_length,
        image_size_hw=image_size_hw,
        device=device,
        dtype=torch.float32,
    )
    num_cond_views = len(image_names)
    cond_rays = torch.zeros(
        1, num_cond_views, 6, image_size_hw[0], image_size_hw[1], device=device
    )
    rays = torch.cat([cond_rays, target_rays], dim=1)
    gt_cond_tokens = torch.cat(
        [
            gt_data["pose_tokens"].cpu(),
            gt_data["scale_tokens"].view(1, 1, 2).cpu().expand(1, num_cond_views, 2),
        ],
        dim=-1,
    )

    vggt_cond_tokens = _build_vggt_condition_tokens(
        image_names=image_names,
        image_size_hw=image_size_hw,
        device=device,
        dtype=dtype,
        mode=mode,
        vggt_checkpoint_path=vggt_checkpoint_path,
    ).cpu()
    zero_cond_tokens = torch.zeros_like(gt_cond_tokens)

    return rays, {
        "unposed": _assemble_cam_tokens(zero_cond_tokens, video_length),
        "vggt_pose": _assemble_cam_tokens(vggt_cond_tokens, video_length),
        "gt_pose": _assemble_cam_tokens(gt_cond_tokens, video_length),
        "debug": {
            "input_poses": gt_data["poses"],
            "shortest_yaw_delta_deg": _shortest_yaw_delta_deg(
                gt_data["poses"][0][0], gt_data["poses"][1][0]
            ),
            "pitch_delta_deg": float(gt_data["poses"][1][1]) - float(gt_data["poses"][0][1]),
            "scale_tokens": gt_data["scale_tokens"].tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="LagerNVS minimal inference")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to 1 or more input images",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=100,
        help="Number of frames to render (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path (default: output_video.mp4)",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="facebook/lagernvs_general_512",
        help="HuggingFace repo ID for the checkpoint when --model_ckpt is not provided",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/share/project/zhouenshen/hpfs/ckpt/diffusion/lagernvs_general_512/model.pt",
        help="Local LagerNVS model checkpoint path; takes precedence over --model_repo",
    )
    parser.add_argument(
        "--vggt_ckpt",
        type=str,
        default="/share/project/zhouenshen/hpfs/ckpt/VGGT-1B/model.pt",
        help="Local VGGT checkpoint path for camera-path estimation",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=["bidirectional_cross_attention", "full_attention"],
        help=(
            "Attention type for the renderer. "
            "Use 'full_attention' for Re10k model, "
            "'bidirectional_cross_attention' for General/DL3DV models."
        ),
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size in pixels (default: 512)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="resize",
        choices=["resize", "square_crop"],
        help=(
            "Image preprocessing mode. "
            "'resize' preserves aspect ratio with longer side = target_size (General model). "
            "'square_crop' center-crops to square then resizes to target_size (256 models)."
        ),
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default="",
        help="Annotation JSON with frame poses and intrinsics; required for GT-based comparisons.",
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help=(
            "Export three videos with a shared GT-interpolated target path: "
            "unposed / VGGT-pose / GT-pose."
        ),
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Device and dtype setup
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 requires Ampere+ GPUs (Compute Capability 8.0+), fall back to float16
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"Device: {device}, dtype: {dtype}")

    # -------------------------------------------------------------------------
    # 2. Load and preprocess input images
    # -------------------------------------------------------------------------
    # load_and_preprocess_images preprocesses input images.
    # "resize" mode: longer side = target_size, aspect ratio preserved (General 512 model).
    # "square_crop" mode: center-crop to square, resize to target_size x target_size (256 models).
    # Returns tensor of shape (num_views, 3, H, W).
    image_names = args.images
    num_cond_views = len(image_names)

    images = load_and_preprocess_images(
        image_names, mode=args.mode, target_size=args.target_size, patch_size=8
    )
    # Add batch dimension: (num_views, 3, H, W) -> (1, num_views, 3, H, W)
    images = images.to(device).unsqueeze(0)
    image_size_hw = (images.shape[-2], images.shape[-1])
    print(f"Loaded {num_cond_views} images, shape: {images.shape}")

    # -------------------------------------------------------------------------
    # 3. Create target camera trajectory
    # -------------------------------------------------------------------------
    # create_target_camera_path uses VGGT (downloaded automatically, ~4GB) to
    # estimate approximate input camera poses, then interpolates a smooth
    # B-spline camera path through them (multi-view) or creates a forward
    # dolly motion (single-view).
    #
    # Returns:
    #   rays:       (1, num_cond_views + video_length, 6, H, W) Plucker ray coords
    #               Conditioning views get zero rays (model doesn't use input poses).
    #   cam_tokens: (1, num_cond_views + video_length, 11) camera tokens encoding
    #               scene scale normalization info.
    compare_outputs = None
    compare_debug = None
    if args.compare_all:
        annotation_json = (
            Path(args.annotation_json).expanduser().resolve()
            if args.annotation_json
            else _infer_annotation_path(image_names)
        )
        print(f"Building GT-interpolated comparison path from: {annotation_json}")
        rays, compare_outputs = _build_compare_inputs(
            image_names=image_names,
            annotation_json=annotation_json,
            image_size_hw=image_size_hw,
            video_length=args.video_length,
            device=device,
            dtype=dtype,
            mode=args.mode,
            vggt_checkpoint_path=args.vggt_ckpt,
        )
        compare_debug = compare_outputs.pop("debug")
        print(
            "GT pair poses:",
            compare_debug["input_poses"],
            "shortest_yaw_delta_deg=",
            compare_debug["shortest_yaw_delta_deg"],
            "pitch_delta_deg=",
            compare_debug["pitch_delta_deg"],
            "scale_tokens=",
            compare_debug["scale_tokens"],
        )
        print(f"Rays shape: {rays.shape}")
    else:
        print("Creating target camera path (downloads VGGT on first run)...")
        rays, cam_tokens = create_target_camera_path(
            image_names,
            args.video_length,
            num_cond_views,
            image_size_hw,
            device,
            dtype,
            mode=args.mode,
            vggt_checkpoint_path=args.vggt_ckpt,
        )
        print(f"Rays shape: {rays.shape}, cam_tokens shape: {cam_tokens.shape}")

    # -------------------------------------------------------------------------
    # 4. Load the LagerNVS model
    # -------------------------------------------------------------------------
    # EncDec_VitB8 = EncoderDecoder with ViT-B/8 config:
    #   - Encoder: VGGT-based feature extractor (pretrained_vggt=False here
    #     because the full model checkpoint already includes trained encoder weights)
    #   - Decoder: 12-layer transformer renderer, patch_size=8, hidden_size=768
    #
    # attention_to_features_type controls how the renderer attends to encoder
    # features:
    #   "bidirectional_cross_attention" — General and DL3DV models
    #   "full_attention"                — Re10k model
    model_ckpt = str(args.model_ckpt).strip() if args.model_ckpt else ""
    if model_ckpt:
        model_ckpt_path = Path(model_ckpt).expanduser().resolve()
        if not model_ckpt_path.is_file():
            raise FileNotFoundError(f"LagerNVS checkpoint not found: {model_ckpt_path}")
        print(f"Loading model from local checkpoint: {model_ckpt_path}")
    else:
        print(f"Loading model from HuggingFace repo: {args.model_repo}")

    model = EncDec_VitB8(
        pretrained_vggt=False,
        attention_to_features_type=args.attention_type,
    )

    if model_ckpt:
        ckpt_path = str(model_ckpt_path)
    else:
        ckpt_path = hf_hub_download(args.model_repo, filename="model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # -------------------------------------------------------------------------
    # 5. Render novel views
    # -------------------------------------------------------------------------
    # render_chunked processes target views in chunks of 16 to manage GPU memory.
    # It internally uses torch.amp.autocast with bfloat16.
    #
    # Input tuple: (cond_images, rays, cam_tokens)
    #   cond_images: (B, num_cond_views, 3, H, W)
    #   rays:        (B, num_cond_views + video_length, 6, H, W)
    #   cam_tokens:  (B, num_cond_views + video_length, 11)
    #
    # Output: (B, video_length, 3, H, W) — rendered RGB frames
    if compare_outputs is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_order = ("unposed", "vggt_pose", "gt_pose")
        for suffix in render_order:
            cam_tokens = compare_outputs[suffix]
            render_path = output_path.with_name(
                f"{output_path.stem}_{suffix}{output_path.suffix}"
            )
            print(f"Rendering {args.video_length} frames for {suffix}...")
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    video_out = render_chunked(
                        model,
                        (images, rays, cam_tokens),
                        num_cond_views=num_cond_views,
                    )
            print(f"Output video shape ({suffix}): {video_out.shape}")
            save_video(video_out[0], str(render_path))
            print(f"Saved to {render_path}")
    else:
        print(f"Rendering {args.video_length} frames...")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                video_out = render_chunked(
                    model,
                    (images, rays, cam_tokens),
                    num_cond_views=num_cond_views,
                )
        print(f"Output video shape: {video_out.shape}")

        # ---------------------------------------------------------------------
        # 6. Save output video
        # ---------------------------------------------------------------------
        save_video(video_out[0], args.output)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
