from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from eval.export import save_video
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vis import compute_plucker_coordinates, render_chunked


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
PREPROCESSING_DIR = REPO_ROOT / "CameraVLM" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from lagernvs_feature_utils import SampleRecord, build_camera_tokens  # noqa: E402


DEFAULT_CAMERA = {
    "width": 512,
    "height": 384,
    "hfov_deg": 90.0,
    "vfov_deg": 67.5,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_dir: str
    input_frames: tuple[str, ...]
    trajectory_kind: str
    pose_modes: tuple[str, ...]
    video_length: int = 73
    frames_per_segment: int = 16
    fps: int = 12


DEFAULT_SCENARIOS = {
    "pap_control_2v": Scenario(
        name="pap_control_2v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_balcony_0025_faucet_0_0_0_0_350_1",
        input_frames=("1.png", "2.png"),
        trajectory_kind="roundtrip",
        pose_modes=("script_auto",),
    ),
    "pap_pitch_mismatch_2v": Scenario(
        name="pap_pitch_mismatch_2v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bathroom_0067_towel_hooks_0_4_236_-19_221_14",
        input_frames=("1.png", "2.png"),
        trajectory_kind="roundtrip",
        pose_modes=("script_auto", "final_target"),
    ),
    "rawpano_pitch_mismatch_2v": Scenario(
        name="rawpano_pitch_mismatch_2v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_485_white_printer_i0_0_266_-20_292_-45",
        input_frames=("1.png", "2.png"),
        trajectory_kind="roundtrip",
        pose_modes=("script_auto", "final_target"),
    ),
    "hstar_4v_polyline": Scenario(
        name="hstar_4v_polyline",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/hstar_sft_512x384_fov_90/camera_image/hstar_sft_hos_104_traj0_Find_the_pizza_f9a1755e",
        input_frames=("1.png", "2.png", "5.png", "6.png"),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=18,
        fps=10,
    ),
    "hstar_6v_polyline": Scenario(
        name="hstar_6v_polyline",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/hstar_sft_512x384_fov_90/camera_image/hstar_sft_hos_104_traj0_Find_the_pizza_f9a1755e",
        input_frames=("1.png", "2.png", "3.png", "4.png", "5.png", "6.png"),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=14,
        fps=10,
    ),
    "pap_bathroom_sink_6v_continuous": Scenario(
        name="pap_bathroom_sink_6v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bathroom_0023_sink_0_4_338_-16_274_-7",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=10,
        fps=12,
    ),
    "pap_bedroom_curtains_6v_continuous": Scenario(
        name="pap_bedroom_curtains_6v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bedroom_0098_curtains_0_5_191_19_296_3",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=10,
        fps=12,
    ),
    "pap_workshop_screen_5v_continuous": Scenario(
        name="pap_workshop_screen_5v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_workshop_0024_large_screen_on_stand_1_5_19_12_287_0",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=10,
        fps=12,
    ),
    "rawpano_hallway_fire_alarm_13v_continuous": Scenario(
        name="rawpano_hallway_fire_alarm_13v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_multi_step_512x384_fov_90/camera_image/RAWPANOMS_2D-3D-S_pano_1_camera_042fab82b3a94af9bea3c80984bc2583_hallway_2_frame_equirectangular_domain_rgb_fire_alarm__rug_12_0_83_3",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=6,
        fps=12,
    ),
    "rawpano_office_trash_window_9v_continuous": Scenario(
        name="rawpano_office_trash_window_9v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_multi_step_512x384_fov_90/camera_image/RAWPANOMS_2D-3D-S_pano_1_camera_04a59ce3e56e4640a6c49d4103089a8e_office_26_frame_equirectangular_domain_rgb_trash_can__window__window_blind_20_0_205_-3",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=6,
        fps=12,
    ),
    "rawpano_structured3d_toilet_11v_continuous": Scenario(
        name="rawpano_structured3d_toilet_11v_continuous",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_multi_step_512x384_fov_90/camera_image/RAWPANOMS_Structured3D_scene_02766_18884_rgb_rawlight_white_door__white_toilet_152590_0_58_16",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=6,
        fps=12,
    ),
    "pap_longest_beaded_cord_9v": Scenario(
        name="pap_longest_beaded_cord_9v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bathroom_0099_beaded_cord_1_1_90_0_167_-45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "pap_longest_curtains_9v": Scenario(
        name="pap_longest_curtains_9v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bedroom_0035_curtains_3_5_250_18_358_-43",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "pap_longest_mop_bucket_8v": Scenario(
        name="pap_longest_mop_bucket_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bathroom_0025_mop_and_bucket_2_1_90_0_316_-45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "pap_longest_beaded_cord_alt_8v": Scenario(
        name="pap_longest_beaded_cord_alt_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_bathroom_0099_beaded_cord_0_3_270_0_167_-45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "pap_longest_kitchen_curtains_8v": Scenario(
        name="pap_longest_kitchen_curtains_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_512x384_fov_90/camera_image/PAP_kitchen_0002_curtains_1_1_90_0_320_-45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "rawpano_longest_matterport_frame_8v": Scenario(
        name="rawpano_longest_matterport_frame_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_1185_black_metal_frame_i0_5_159_-20_316_45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "rawpano_longest_matterport_arch_8v": Scenario(
        name="rawpano_longest_matterport_arch_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_3881_wooden_decorative_arch_i0_5_237_-18_73_45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "rawpano_longest_matterport_window_8v": Scenario(
        name="rawpano_longest_matterport_window_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_4142_black_window_frames_i0_5_123_20_357_-45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "rawpano_longest_matterport_door_8v": Scenario(
        name="rawpano_longest_matterport_door_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_4171_white_door_frame_i0_4_267_-11_147_45",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
    "rawpano_longest_matterport_door_alt_8v": Scenario(
        name="rawpano_longest_matterport_door_alt_8v",
        seq_dir="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image/RAWPANO_Matterport3D_6327_white_door_frame_i0_5_46_-16_257_44",
        input_frames=(),
        trajectory_kind="polyline",
        pose_modes=("script_auto",),
        frames_per_segment=8,
        fps=12,
    ),
}


def _rotation_x(angle_rad: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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


def _rotation_y(angle_rad: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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


def _build_c2w_from_yaw_pitch(
    yaw_deg: float,
    pitch_deg: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(pitch_deg))
    rotation = _rotation_y(yaw_rad, device, dtype) @ _rotation_x(
        pitch_rad, device, dtype
    )
    extrinsic = torch.zeros(4, 4, device=device, dtype=dtype)
    extrinsic[:3, :3] = rotation
    extrinsic[3, 3] = 1.0
    return extrinsic


def _shortest_yaw_delta_deg(src_yaw: float, dst_yaw: float) -> float:
    return ((float(dst_yaw) - float(src_yaw) + 180.0) % 360.0) - 180.0


def _normalize_c2w_for_lagernvs(
    c2w: torch.Tensor,
    num_cond_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _assemble_cam_tokens(cond_tokens: torch.Tensor, video_length: int) -> torch.Tensor:
    batch_size, num_cond_views, _ = cond_tokens.shape
    total_views = num_cond_views + video_length
    cam_tokens = torch.zeros(batch_size, total_views, 11, dtype=torch.float32)
    cam_tokens[:, :num_cond_views, :] = cond_tokens.float().cpu()
    cam_tokens[:, :, 9:] = cond_tokens[:, 0:1, 9:].float().cpu().expand(
        batch_size, total_views, 2
    )
    return cam_tokens


def _camera_params_from_metadata(camera: dict[str, float]) -> tuple[float, float, float, float, int, int]:
    width = int(round(float(camera.get("width", DEFAULT_CAMERA["width"]))))
    height = int(round(float(camera.get("height", DEFAULT_CAMERA["height"]))))

    fx = camera.get("fx")
    fy = camera.get("fy")
    cx = camera.get("cx")
    cy = camera.get("cy")
    if None not in (fx, fy, cx, cy):
        return float(fx), float(fy), float(cx), float(cy), width, height

    hfov_deg = float(camera.get("hfov_deg", DEFAULT_CAMERA["hfov_deg"]))
    vfov_deg = float(camera.get("vfov_deg", DEFAULT_CAMERA["vfov_deg"]))
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy, width, height


def _default_camera_dict(annotation: dict[str, object]) -> dict[str, float]:
    camera = dict(DEFAULT_CAMERA)
    anno_camera = annotation.get("camera")
    if isinstance(anno_camera, dict):
        camera.update(anno_camera)
    return camera


def _sorted_frame_names(seq_dir: Path) -> list[str]:
    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 10**9, path.name

    return [path.name for path in sorted(seq_dir.glob("*.png"), key=sort_key)]


def _load_annotation(seq_dir: Path) -> dict[str, object]:
    with open(seq_dir / "annotation.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pose_from_frames(
    annotation: dict[str, object],
    frame_name: str,
) -> tuple[float, float] | None:
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        return None
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_name = Path(str(frame.get("image", ""))).name
        if image_name != frame_name:
            continue
        pose = frame.get("pose")
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
        if "yaw" in frame and "pitch" in frame:
            return float(frame["yaw"]), float(frame["pitch"])
    return None


def _resolve_pose_from_pose_list(
    annotation: dict[str, object],
    frame_name: str,
) -> tuple[float, float] | None:
    poses = annotation.get("poses")
    if not isinstance(poses, list):
        return None
    frame_idx = int(Path(frame_name).stem) - 1
    if 0 <= frame_idx < len(poses):
        pose = poses[frame_idx]
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
    return None


def _resolve_pose_from_images_and_poses(
    annotation: dict[str, object],
    frame_name: str,
) -> tuple[float, float] | None:
    images = annotation.get("images")
    poses = annotation.get("poses")
    if not isinstance(images, list) or not isinstance(poses, list):
        return None
    for image_name, pose in zip(images, poses):
        if Path(str(image_name)).name != frame_name:
            continue
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
    return None


def _resolve_pose_from_actions(
    annotation: dict[str, object],
    frame_name: str,
    *,
    total_frames: int,
    use_final_target_for_last_frame: bool,
) -> tuple[float, float]:
    initial_yaw = float(annotation.get("initial_yaw", 0.0))
    initial_pitch = float(annotation.get("initial_pitch", 0.0))
    frame_index = int(Path(frame_name).stem)

    if (
        use_final_target_for_last_frame
        and frame_index == total_frames
        and "target_yaw" in annotation
        and "target_pitch" in annotation
    ):
        return float(annotation["target_yaw"]) % 360.0, float(annotation["target_pitch"])

    yaw = initial_yaw
    pitch = initial_pitch
    actions = annotation.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    for action in actions[: max(0, frame_index - 1)]:
        if not isinstance(action, (list, tuple)) or len(action) < 2:
            continue
        yaw += float(action[0])
        pitch += float(action[1])
    return yaw % 360.0, float(pitch)


def resolve_poses(
    annotation: dict[str, object],
    frame_names: Sequence[str],
    *,
    total_frames: int,
    pose_mode: str,
) -> list[tuple[float, float]]:
    if pose_mode not in {"script_auto", "final_target"}:
        raise ValueError(f"Unsupported pose mode: {pose_mode}")

    poses: list[tuple[float, float]] = []
    use_final_target_for_last_frame = pose_mode == "final_target"
    for frame_name in frame_names:
        for resolver in (
            _resolve_pose_from_frames,
            _resolve_pose_from_images_and_poses,
            _resolve_pose_from_pose_list,
        ):
            pose = resolver(annotation, frame_name)
            if pose is not None:
                poses.append(pose)
                break
        else:
            poses.append(
                _resolve_pose_from_actions(
                    annotation,
                    frame_name,
                    total_frames=total_frames,
                    use_final_target_for_last_frame=use_final_target_for_last_frame,
                )
            )
    return poses


def build_condition_tokens(
    poses: Sequence[tuple[float, float]],
    camera: dict[str, float],
    image_size_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fx, fy, cx, cy, src_w, src_h = _camera_params_from_metadata(camera)
    tgt_h, tgt_w = image_size_hw
    scale_x = float(tgt_w) / float(src_w)
    scale_y = float(tgt_h) / float(src_h)
    intrinsics_single = torch.tensor(
        [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y],
        device=device,
        dtype=torch.float32,
    )
    intrinsics = intrinsics_single.unsqueeze(0).repeat(len(poses), 1)

    c2w = torch.stack(
        [
            _build_c2w_from_yaw_pitch(yaw, pitch, device=device, dtype=torch.float32)
            for yaw, pitch in poses
        ],
        dim=0,
    ).unsqueeze(0)
    c2w, scale_tokens = _normalize_c2w_for_lagernvs(c2w, num_cond_views=len(poses))
    pose_tokens = extri_intri_to_pose_encoding(
        c2w[:, :, :3, :],
        intrinsics.unsqueeze(0),
        image_size_hw=image_size_hw,
    ).float()
    cond_tokens = torch.cat(
        [
            pose_tokens.cpu(),
            scale_tokens.view(1, 1, 2).cpu().expand(1, len(poses), 2),
        ],
        dim=-1,
    )
    return cond_tokens, intrinsics.unsqueeze(0)


def _interpolate_pose(
    src_pose: tuple[float, float],
    dst_pose: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    yaw = float(src_pose[0]) + _shortest_yaw_delta_deg(src_pose[0], dst_pose[0]) * float(t)
    pitch = float(src_pose[1]) + (float(dst_pose[1]) - float(src_pose[1])) * float(t)
    return yaw % 360.0, pitch


def build_roundtrip_trajectory(
    poses: Sequence[tuple[float, float]],
    video_length: int,
) -> tuple[list[tuple[float, float]], list[int], list[int]]:
    if len(poses) != 2:
        raise ValueError("Roundtrip trajectory requires exactly 2 conditioning poses.")
    mid_idx = video_length // 2
    forward_ts = np.linspace(0.0, 1.0, mid_idx + 1, dtype=np.float32)
    backward_ts = np.linspace(0.0, 1.0, video_length - mid_idx, dtype=np.float32)[1:]
    forward = [_interpolate_pose(poses[0], poses[1], float(t)) for t in forward_ts]
    backward = [_interpolate_pose(poses[1], poses[0], float(t)) for t in backward_ts]
    trajectory = forward + backward
    anchor_indices = [0, mid_idx, len(trajectory) - 1]
    reference_indices = [0, 1, 0]
    return trajectory, anchor_indices, reference_indices


def build_polyline_trajectory(
    poses: Sequence[tuple[float, float]],
    frames_per_segment: int,
) -> tuple[list[tuple[float, float]], list[int], list[int]]:
    if len(poses) < 2:
        raise ValueError("Polyline trajectory requires at least 2 conditioning poses.")
    trajectory: list[tuple[float, float]] = []
    anchor_indices = [0]
    for segment_idx in range(len(poses) - 1):
        ts = np.linspace(0.0, 1.0, frames_per_segment + 1, dtype=np.float32)
        segment = [
            _interpolate_pose(poses[segment_idx], poses[segment_idx + 1], float(t))
            for t in ts
        ]
        if segment_idx > 0:
            segment = segment[1:]
        trajectory.extend(segment)
        anchor_indices.append(len(trajectory) - 1)
    reference_indices = list(range(len(poses)))
    return trajectory, anchor_indices, reference_indices


def build_target_rays(
    target_poses: Sequence[tuple[float, float]],
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    target_c2w = torch.stack(
        [
            _build_c2w_from_yaw_pitch(yaw, pitch, device=device, dtype=torch.float32)
            for yaw, pitch in target_poses
        ],
        dim=0,
    ).unsqueeze(0)
    first_cam_inv = torch.linalg.inv(target_c2w[:, 0:1, :, :])
    target_c2w = first_cam_inv @ target_c2w
    target_fxfycxcy = intrinsics[:, 0:1, :].expand(1, len(target_poses), 4).contiguous()
    return compute_plucker_coordinates(target_c2w, target_fxfycxcy, image_size_hw)


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image = np.transpose(image, (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def _save_keyframe_panel(
    panel_path: Path,
    rendered_video: torch.Tensor,
    reference_images: torch.Tensor,
    anchor_indices: Sequence[int],
    reference_indices: Sequence[int],
) -> list[dict[str, float]]:
    rows: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    for frame_idx, ref_idx in zip(anchor_indices, reference_indices):
        pred = rendered_video[frame_idx]
        ref = reference_images[ref_idx]
        diff = (pred - ref).abs()
        mse = float(torch.mean((pred - ref) ** 2).item())
        mae = float(torch.mean(diff).item())
        psnr = float(-10.0 * math.log10(max(mse, 1e-8)))
        metrics.append(
            {
                "video_frame_index": int(frame_idx),
                "reference_input_index": int(ref_idx),
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
            }
        )
        row = np.concatenate(
            [
                _tensor_to_uint8(ref),
                _tensor_to_uint8(pred),
                _tensor_to_uint8(torch.clamp(diff * 4.0, 0.0, 1.0)),
            ],
            axis=1,
        )
        rows.append(row)
    panel = np.concatenate(rows, axis=0)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(panel_path)
    return metrics


def _load_model(
    checkpoint_path: str,
    attention_type: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = EncDec_VitB8(
        pretrained_vggt=False,
        attention_to_features_type=attention_type,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def render_scenario(
    model: torch.nn.Module,
    scenario: Scenario,
    pose_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    output_root: Path,
    frames_per_segment_scale: int,
) -> dict[str, object]:
    seq_dir = Path(scenario.seq_dir)
    annotation = _load_annotation(seq_dir)
    all_frame_names = _sorted_frame_names(seq_dir)
    input_frames = tuple(scenario.input_frames) if scenario.input_frames else tuple(all_frame_names)
    image_names = [str(seq_dir / frame) for frame in input_frames]

    images = load_and_preprocess_images(
        image_names,
        mode="resize",
        target_size=512,
        patch_size=8,
    )
    image_size_hw = (int(images.shape[-2]), int(images.shape[-1]))
    input_images = images.unsqueeze(0).to(device)
    if device.type == "cuda":
        input_images = input_images.to(dtype=dtype)

    extract_script_poses = resolve_poses(
        annotation,
        input_frames,
        total_frames=len(all_frame_names),
        pose_mode="script_auto",
    )
    selected_poses = resolve_poses(
        annotation,
        input_frames,
        total_frames=len(all_frame_names),
        pose_mode=pose_mode,
    )
    camera = _default_camera_dict(annotation)

    cond_tokens, intrinsics = build_condition_tokens(
        selected_poses,
        camera,
        image_size_hw,
        device,
    )
    extract_tokens, _ = build_condition_tokens(
        extract_script_poses,
        camera,
        image_size_hw,
        device,
    )

    record = SampleRecord(
        metadata_index=0,
        sample_key=f"{scenario.name}_{pose_mode}",
        image_rel_paths=tuple(input_frames),
        image_abs_paths=tuple(image_names),
        poses=tuple(selected_poses),
        camera=camera,
        output_path="/tmp/unused.npz",
    )
    extracted_style_tokens = build_camera_tokens(record, image_size_hw, "metadata_pose_world_scale")
    build_camera_tokens_max_abs_diff = float((cond_tokens - extracted_style_tokens).abs().max().item())
    selected_vs_extract_cam_token_max_abs_diff = float((cond_tokens - extract_tokens).abs().max().item())

    if scenario.trajectory_kind == "roundtrip":
        target_poses, anchor_indices, reference_indices = build_roundtrip_trajectory(
            selected_poses,
            scenario.video_length,
        )
    elif scenario.trajectory_kind == "polyline":
        effective_frames_per_segment = max(
            1, int(scenario.frames_per_segment) * max(1, int(frames_per_segment_scale))
        )
        target_poses, anchor_indices, reference_indices = build_polyline_trajectory(
            selected_poses,
            effective_frames_per_segment,
        )
    else:
        raise ValueError(f"Unsupported trajectory kind: {scenario.trajectory_kind}")

    target_rays = build_target_rays(target_poses, intrinsics, image_size_hw, device)
    cond_rays = torch.zeros(
        1,
        len(input_frames),
        6,
        image_size_hw[0],
        image_size_hw[1],
        device=device,
    )
    rays = torch.cat([cond_rays, target_rays], dim=1)
    cam_tokens = _assemble_cam_tokens(cond_tokens, len(target_poses))

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            video_out = render_chunked(
                model,
                (input_images, rays, cam_tokens),
                num_cond_views=len(input_frames),
            )
    video_out = video_out[0].detach().float().cpu()

    scenario_dir = output_root / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    video_path = scenario_dir / f"{pose_mode}.mp4"
    panel_path = scenario_dir / f"{pose_mode}_keyframes.png"
    metrics_path = scenario_dir / f"{pose_mode}_metrics.json"
    save_video(video_out, str(video_path), fps=scenario.fps)
    anchor_metrics = _save_keyframe_panel(
        panel_path=panel_path,
        rendered_video=video_out,
        reference_images=images,
        anchor_indices=anchor_indices,
        reference_indices=reference_indices,
    )

    metrics = {
        "scenario": scenario.name,
        "pose_mode": pose_mode,
        "seq_dir": str(seq_dir),
        "input_frames": list(input_frames),
        "selected_poses": [[float(yaw), float(pitch)] for yaw, pitch in selected_poses],
        "extract_script_poses": [
            [float(yaw), float(pitch)] for yaw, pitch in extract_script_poses
        ],
        "trajectory_kind": scenario.trajectory_kind,
        "trajectory_length": len(target_poses),
        "frames_per_segment_scale": int(frames_per_segment_scale),
        "effective_frames_per_segment": (
            effective_frames_per_segment if scenario.trajectory_kind == "polyline" else None
        ),
        "trajectory_first_pose": list(target_poses[0]),
        "trajectory_mid_pose": list(target_poses[len(target_poses) // 2]),
        "trajectory_last_pose": list(target_poses[-1]),
        "anchor_indices": [int(idx) for idx in anchor_indices],
        "reference_indices": [int(idx) for idx in reference_indices],
        "keyframe_metrics": anchor_metrics,
        "build_camera_tokens_max_abs_diff": build_camera_tokens_max_abs_diff,
        "selected_vs_extract_cam_token_max_abs_diff": selected_vs_extract_cam_token_max_abs_diff,
        "video_path": str(video_path),
        "keyframe_panel_path": str(panel_path),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT-pose validation videos for LagerNVS.")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        help=f"Scenario names or 'all'. Available: {', '.join(sorted(DEFAULT_SCENARIOS))}",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(REPO_ROOT / "_debug_runs" / "gt_pose_validation"),
        help="Directory for videos, panels, and metrics.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/share/project/zhouenshen/hpfs/ckpt/diffusion/lagernvs_general_512/model.pt",
        help="Local LagerNVS checkpoint path.",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=["bidirectional_cross_attention", "full_attention"],
        help="Renderer attention type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--frames_per_segment_scale",
        type=int,
        default=1,
        help="Multiplier for polyline interpolation density. Keeps original files untouched when used with a new output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = sorted(DEFAULT_SCENARIOS) if "all" in args.scenario else args.scenario
    scenarios = [DEFAULT_SCENARIOS[name] for name in scenario_names]

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device_arg = args.device
    if device_arg == "cuda" and torch.cuda.is_available():
        device_arg = "cuda:0"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        device_arg = "cpu"
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"Using device={device}, dtype={dtype}")
    print(f"Loading model from {args.model_ckpt}")
    model = _load_model(args.model_ckpt, args.attention_type, device)

    summary: list[dict[str, object]] = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario.name}")
        for pose_mode in scenario.pose_modes:
            print(f"  pose_mode={pose_mode}")
            metrics = render_scenario(
                model=model,
                scenario=scenario,
                pose_mode=pose_mode,
                device=device,
                dtype=dtype,
                output_root=output_root,
                frames_per_segment_scale=args.frames_per_segment_scale,
            )
            summary.append(metrics)
            psnr_values = [item["psnr"] for item in metrics["keyframe_metrics"]]
            print(
                "    saved",
                metrics["video_path"],
                "psnr=",
                [round(float(value), 3) for value in psnr_values],
                "token_diff_vs_extract=",
                round(float(metrics["selected_vs_extract_cam_token_max_abs_diff"]), 6),
            )

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
