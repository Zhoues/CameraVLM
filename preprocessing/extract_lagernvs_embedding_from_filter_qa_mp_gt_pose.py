"""
/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_filter_qa_mp_gt_pose.py \
  --dataset_name pap_filter_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1 \
  --workers_per_gpu 4

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_filter_qa_mp_gt_pose.py \
  --dataset_name raw_pano_filter_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0,1,2,3 \
  --workers_per_gpu 4 \
  --num_machines 2 \
  --machine_rank 0

/share/project/zhouenshen/miniconda3/envs/anno/bin/python \
/share/project/zhouenshen/hpfs/code/ActivePerception/CameraVLM/preprocessing/extract_lagernvs_embedding_from_filter_qa_mp_gt_pose.py \
  --dataset_name hstar_sft_ours_filter_512x384_fov_90 \
  --use_gt_pose \
  --gpus 0 \
  --workers_per_gpu 1 \
  --max_samples 128 \
  --force_rebuild_manifest \
  --overwrite
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import extract_lagernvs_embedding_from_qa_mp_gt_pose as BASE  # noqa: E402
import lagernvs_feature_utils as LF  # noqa: E402


FILTER_DATASET_NAMES = {
    "pap_filter_512x384_fov_90",
    "raw_pano_filter_512x384_fov_90",
    "hstar_sft_ours_filter_512x384_fov_90",
}
IMAGE_TOKEN_PATTERN = re.compile(r"^(?P<prefix>\s*)(?P<tokens>(?:<image>\s*)+)")

BASE_VALIDATE_OUTPUT_NPZ = LF.validate_output_npz
BASE_EXTRACT_SINGLE_RECORD = LF.extract_single_record


@dataclass(frozen=True)
class FilterMemoryRecord:
    metadata_index: int
    sample_key: str
    image_rel_paths: tuple[str, ...]
    image_abs_paths: tuple[str, ...]
    poses: tuple[tuple[float, float], ...]
    camera: dict[str, Any]
    output_path: str
    filtered_frame_ids: tuple[int, ...]
    observed_image_rel_paths: tuple[str, ...]
    prefix_frame_count: int
    sequence_dir_rel_path: str
    annotation_path: str
    source_sample_id: str


def _infer_dataset_name_from_qa_path(qa_input_path: str | Path) -> str:
    return Path(qa_input_path).resolve().parent.name


def _infer_camera_image_root_from_qa_path(qa_input_path: str | Path) -> Path:
    qa_input_path = Path(qa_input_path).resolve()
    dataset_name = _infer_dataset_name_from_qa_path(qa_input_path)

    try:
        resolved = LF.resolve_qa_latent_paths(dataset_name=dataset_name)
        candidate = Path(resolved.image_root)
        if candidate.is_dir():
            return candidate
    except Exception:
        pass

    dataset_dir = qa_input_path.parent
    if dataset_dir.parent.name == "vlm_data":
        candidate = dataset_dir.parent.parent / "camera_data" / dataset_dir.name / "camera_image"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(f"failed to infer camera_image_root from {qa_input_path}")


def _normalize_rel_path(path_text: str) -> str:
    path = str(path_text or "").strip().replace("\\", "/")
    path = path.lstrip("/")
    path = str(Path(path))
    if path in {"", "."}:
        return ""
    if path.startswith(".."):
        raise ValueError(f"relative path escapes root: {path_text}")
    return path.replace("\\", "/")


def _rewrite_user_image_placeholders(text: str, num_images: int) -> str:
    if not isinstance(text, str):
        return text
    num_images = max(0, int(num_images))
    match = IMAGE_TOKEN_PATTERN.match(text)
    if match is None:
        return text
    rebuilt_tokens = "<image>" * num_images
    return f"{match.group('prefix')}{rebuilt_tokens}{text[match.end():]}"


def _shortest_yaw_delta_deg(src_yaw: float, dst_yaw: float) -> float:
    return ((float(dst_yaw) - float(src_yaw) + 180.0) % 360.0) - 180.0


def _pose_to_c2w_4x4(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    c2w_3x4 = LF._build_c2w_from_yaw_pitch(yaw_deg, pitch_deg)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = c2w_3x4
    return c2w


def _adjacent_pose_deltas(poses: tuple[tuple[float, float], ...]) -> np.ndarray:
    if len(poses) <= 1:
        return np.zeros((0, 2), dtype=np.float32)
    deltas = []
    for (yaw0, pitch0), (yaw1, pitch1) in zip(poses[:-1], poses[1:]):
        deltas.append(
            [
                float(_shortest_yaw_delta_deg(yaw0, yaw1)),
                float(pitch1 - pitch0),
            ]
        )
    return np.asarray(deltas, dtype=np.float32)


def _build_relative_extrinsics(
    poses: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not poses:
        empty = np.zeros((0, 4, 4), dtype=np.float32)
        return empty, empty, empty

    c2w = np.stack([_pose_to_c2w_4x4(yaw, pitch) for yaw, pitch in poses], axis=0).astype(np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    if len(poses) <= 1:
        empty = np.zeros((0, 4, 4), dtype=np.float32)
        return c2w, w2c, empty
    prev_to_cur = np.matmul(w2c[1:], c2w[:-1]).astype(np.float32)
    return c2w, w2c, prev_to_cur


def _load_annotation_payload(
    annotation_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, str]]:
    payload, _frame_lookup, _image_pose_lookup = BASE._load_sequence_metadata(annotation_path)
    frame_by_id: dict[int, dict[str, Any]] = {}
    frame_id_to_image: dict[int, str] = {}
    frames = payload.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_id = frame.get("frame_id")
            try:
                frame_id_int = int(frame_id)
            except (TypeError, ValueError):
                continue
            frame_by_id[frame_id_int] = frame
            image_rel = _normalize_rel_path(str(frame.get("image", "")))
            if image_rel:
                frame_id_to_image[frame_id_int] = image_rel
    return payload, frame_by_id, frame_id_to_image


def _prefix_state_for_count(
    annotation_payload: dict[str, Any],
    prefix_count: int,
) -> tuple[dict[str, Any], int]:
    prefix_states = annotation_payload.get("prefix_states", [])
    if not isinstance(prefix_states, list):
        raise KeyError("prefix_states missing from filter annotation")
    available_counts: list[int] = []
    for state_idx, prefix_state in enumerate(prefix_states):
        if not isinstance(prefix_state, dict):
            continue
        state_prefix_count = int(prefix_state.get("prefix_frame_count", -1))
        if state_prefix_count > 0:
            available_counts.append(state_prefix_count)
        if state_prefix_count == int(prefix_count):
            return prefix_state, state_idx
    if available_counts:
        capped_prefix_count = max(count for count in available_counts if count <= max(1, int(prefix_count)))
        for state_idx, prefix_state in enumerate(prefix_states):
            if not isinstance(prefix_state, dict):
                continue
            if int(prefix_state.get("prefix_frame_count", -1)) == int(capped_prefix_count):
                return prefix_state, state_idx
    raise KeyError(f"prefix_state not found for prefix_count={prefix_count}")


def _retained_sequence_from_sample(
    *,
    sample_images: list[str],
    annotation_payload: dict[str, Any],
    frame_by_id: dict[int, dict[str, Any]],
    frame_id_to_image: dict[int, str],
    annotation_path: Path,
) -> tuple[list[str], tuple[tuple[float, float], ...], tuple[int, ...], int]:
    if not sample_images:
        raise ValueError(f"empty images in sample for {annotation_path}")

    requested_prefix_count = len(sample_images)
    prefix_count = requested_prefix_count
    prefix_state, _state_idx = _prefix_state_for_count(annotation_payload, prefix_count)
    prefix_count = int(prefix_state.get("prefix_frame_count", prefix_count))

    retained_ids_raw = prefix_state.get("retained_frame_ids", [])
    retained_ids = tuple(int(frame_id) for frame_id in retained_ids_raw if int(frame_id) > 0)
    if not retained_ids:
        retained_ids = tuple(range(1, prefix_count + 1))

    filtered_image_refs: list[str] = []
    filtered_poses: list[tuple[float, float]] = []
    for frame_id in retained_ids:
        image_rel = frame_id_to_image.get(frame_id)
        if not image_rel:
            first_sample = Path(str(sample_images[0]))
            seq_dir_rel = str(first_sample.parent) if first_sample.parent != Path(".") else BASE._derive_sequence_dir_rel_path(str(first_sample))
            image_rel = _normalize_rel_path(f"{seq_dir_rel}/{frame_id}.png")

        frame = frame_by_id.get(frame_id)
        if frame is not None:
            pose = BASE._extract_pose_from_annotation_frame(
                frame=frame,
                annotation_path=annotation_path,
                frame_name=Path(image_rel).name,
            )
        else:
            pose = BASE._resolve_pose_for_image(
                image_rel_path=image_rel,
                payload=annotation_payload,
                frame_lookup={},
                image_pose_lookup={},
                annotation_path=annotation_path,
            )

        filtered_image_refs.append(image_rel)
        filtered_poses.append((float(pose[0]), float(pose[1])))

    return filtered_image_refs, tuple(filtered_poses), retained_ids, prefix_count


def build_filter_memory_latent_manifest(
    qa_input_path: str | Path,
    qa_output_path: str | Path,
    sequence_manifest_path: str | Path,
    sequence_manifest_dir: str | Path,
    num_sequence_shards: int,
    show_progress: bool = True,
    progress_desc: str = "Scanning filter QA",
    max_samples: int | None = None,
) -> dict[str, Any]:
    qa_input_path = str(qa_input_path)
    qa_output_path = str(qa_output_path)
    sequence_manifest_path = Path(sequence_manifest_path)
    shard_paths = LF.reset_sequence_manifest_dir(sequence_manifest_dir, num_sequence_shards)
    sequence_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    camera_image_root = _infer_camera_image_root_from_qa_path(qa_input_path)
    dataset_name = _infer_dataset_name_from_qa_path(qa_input_path)

    digest_to_name: dict[str, str] = {}
    annotation_cache: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, str]]] = {}
    shard_files = [open(path, "w", encoding="utf-8") for path in shard_paths]
    main_manifest_file = open(sequence_manifest_path, "w", encoding="utf-8")

    total_samples = 0
    samples_with_images = 0
    empty_image_samples = 0
    unique_sequences = 0
    duplicate_sequences = 0
    rewritten_placeholder_samples = 0
    truncated_prefix_samples = 0
    shard_counts = [0 for _ in shard_paths]
    retained_view_count_hist: dict[int, int] = {}
    try:
        with LF.JsonArrayWriter(qa_output_path) as qa_writer:
            iterator = LF.iter_json_array(
                qa_input_path,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
            qa_bar = LF.tqdm(
                total=(None if max_samples is None else int(max_samples)),
                desc=f"{progress_desc} qa",
                unit="sample",
                **LF.tqdm_options(disable=not show_progress),
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
                        observed_image_refs = [_normalize_rel_path(str(img_ref)) for img_ref in images]
                        sequence_dir_rel = BASE._derive_sequence_dir_rel_path(observed_image_refs[0])
                        sequence_dir_path = camera_image_root / sequence_dir_rel
                        annotation_path = sequence_dir_path / "annotation.json"
                        if not annotation_path.is_file():
                            raise FileNotFoundError(f"annotation.json not found: {annotation_path}")

                        annotation_key = str(annotation_path)
                        annotation_payload, frame_by_id, frame_id_to_image = annotation_cache.get(
                            annotation_key,
                            ({}, {}, {}),
                        )
                        if not annotation_payload:
                            annotation_payload, frame_by_id, frame_id_to_image = _load_annotation_payload(annotation_path)
                            annotation_cache[annotation_key] = (
                                annotation_payload,
                                frame_by_id,
                                frame_id_to_image,
                            )

                        filtered_image_refs, filtered_poses, filtered_frame_ids, prefix_count = _retained_sequence_from_sample(
                            sample_images=observed_image_refs,
                            annotation_payload=annotation_payload,
                            frame_by_id=frame_by_id,
                            frame_id_to_image=frame_id_to_image,
                            annotation_path=annotation_path,
                        )
                        if prefix_count < len(observed_image_refs):
                            observed_image_refs = observed_image_refs[:prefix_count]
                            truncated_prefix_samples += 1
                        if not filtered_image_refs:
                            empty_image_samples += 1
                            item_dict["latents"] = latent_names
                            qa_writer.write(item_dict)
                            qa_bar.update(1)
                            continue

                        digest = LF.hash_image_sequence(filtered_image_refs)
                        latent_name = digest_to_name.get(digest)
                        if latent_name is None:
                            latent_idx = unique_sequences
                            latent_name = f"{latent_idx:09d}.npz"
                            digest_to_name[digest] = latent_name
                            unique_sequences += 1

                            annotation_camera = annotation_payload.get("camera")
                            camera_dict = dict(BASE.DEFAULT_CAMERA)
                            if isinstance(annotation_camera, dict):
                                camera_dict.update(annotation_camera)

                            sequence_line = {
                                "latent_idx": latent_idx,
                                "latent_name": latent_name,
                                "sequence_hash": digest,
                                "num_views": len(filtered_image_refs),
                                "image_rel_paths": filtered_image_refs,
                                "poses": [[float(yaw), float(pitch)] for yaw, pitch in filtered_poses],
                                "camera": camera_dict,
                                "filtered_frame_ids": [int(frame_id) for frame_id in filtered_frame_ids],
                                "observed_image_rel_paths": observed_image_refs,
                                "prefix_frame_count": int(prefix_count),
                                "sequence_dir_rel_path": sequence_dir_rel,
                                "annotation_path": str(annotation_path),
                                "source_sample_id": str(item_dict.get("id", "")),
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
                        retained_count = len(filtered_image_refs)
                        retained_view_count_hist[retained_count] = retained_view_count_hist.get(retained_count, 0) + 1

                        item_out = dict(item_dict)
                        item_out["observed_images"] = observed_image_refs
                        item_out["images"] = filtered_image_refs
                        item_out["filtered_images"] = filtered_image_refs
                        item_out["retained_frame_ids"] = [int(frame_id) for frame_id in filtered_frame_ids]
                        item_out["prefix_frame_count"] = int(prefix_count)
                        item_out["sequence_dir"] = sequence_dir_rel
                        item_out["latents"] = latent_names

                        conversations = item_out.get("conversations")
                        if isinstance(conversations, list):
                            rewritten = []
                            rewritten_any = False
                            for turn in conversations:
                                if not isinstance(turn, dict):
                                    rewritten.append(turn)
                                    continue
                                turn_dict = dict(turn)
                                if turn_dict.get("from") == "human" and isinstance(turn_dict.get("value"), str):
                                    new_value = _rewrite_user_image_placeholders(
                                        turn_dict["value"],
                                        len(filtered_image_refs),
                                    )
                                    rewritten_any = rewritten_any or (new_value != turn_dict["value"])
                                    turn_dict["value"] = new_value
                                rewritten.append(turn_dict)
                            if rewritten_any:
                                rewritten_placeholder_samples += 1
                            item_out["conversations"] = rewritten
                        qa_writer.write(item_out)
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

    LF.log_info(
        f"filter-memory QA scan finished for dataset={dataset_name}: "
        f"total_samples={total_samples}, unique_sequences={unique_sequences}, "
        f"duplicates={duplicate_sequences}, empty_image_samples={empty_image_samples}"
    )
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
        "rewritten_placeholder_samples": rewritten_placeholder_samples,
        "truncated_prefix_samples": truncated_prefix_samples,
        "retained_view_count_hist": dict(sorted(retained_view_count_hist.items())),
        "camera_image_root": str(camera_image_root),
        "shard_counts": shard_counts,
    }


def _load_filter_sample_records_from_sequence_manifest_with_gt_pose(
    manifest_path: str | Path,
    image_root: str | Path,
    latent_dir: str | Path,
    camera_image_root: str | Path | None,
    default_camera: dict[str, Any] | None = None,
) -> list[FilterMemoryRecord]:
    del camera_image_root
    image_root = str(image_root)
    latent_dir = str(latent_dir)
    default_camera = dict(default_camera or {})
    records: list[FilterMemoryRecord] = []
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
                path if Path(path).is_absolute() else str(Path(image_root) / path)
                for path in image_rel_paths
            )
            latent_idx = int(item.get("latent_idx", len(records)))
            latent_name = str(item.get("latent_name", f"{latent_idx:09d}.npz"))
            sample_key = Path(latent_name).stem

            poses_raw = item.get("poses", [])
            poses = tuple(
                (float(pose[0]), float(pose[1]))
                for pose in poses_raw
                if isinstance(pose, (list, tuple)) and len(pose) >= 2
            )
            if len(poses) != len(image_rel_paths):
                raise ValueError(
                    f"poses/image_rel_paths length mismatch in manifest {manifest_path}: "
                    f"{len(poses)} vs {len(image_rel_paths)}"
                )

            camera = item.get("camera")
            if not isinstance(camera, dict):
                camera = dict(default_camera)
            else:
                camera = {**default_camera, **camera}

            records.append(
                FilterMemoryRecord(
                    metadata_index=latent_idx,
                    sample_key=sample_key,
                    image_rel_paths=image_rel_paths,
                    image_abs_paths=image_abs_paths,
                    poses=poses,
                    camera=camera,
                    output_path=str(Path(latent_dir) / latent_name),
                    filtered_frame_ids=tuple(int(x) for x in item.get("filtered_frame_ids", [])),
                    observed_image_rel_paths=tuple(str(x) for x in item.get("observed_image_rel_paths", [])),
                    prefix_frame_count=int(item.get("prefix_frame_count", len(item.get("observed_image_rel_paths", [])))),
                    sequence_dir_rel_path=str(item.get("sequence_dir_rel_path", "")),
                    annotation_path=str(item.get("annotation_path", "")),
                    source_sample_id=str(item.get("source_sample_id", "")),
                )
            )
    return records


def _augment_filter_payload(
    payload: dict[str, Any],
    record: FilterMemoryRecord,
) -> dict[str, Any]:
    c2w, w2c, prev_to_cur = _build_relative_extrinsics(record.poses)
    adjacent_delta_yaw_pitch = _adjacent_pose_deltas(record.poses)
    payload.update(
        {
            "filtered_frame_ids": np.asarray(record.filtered_frame_ids, dtype=np.int32),
            "observed_image_rel_paths": np.asarray(record.observed_image_rel_paths),
            "prefix_frame_count": np.asarray([record.prefix_frame_count], dtype=np.int32),
            "sequence_dir_rel_path": np.asarray(record.sequence_dir_rel_path),
            "annotation_path": np.asarray(record.annotation_path),
            "source_sample_id": np.asarray(record.source_sample_id),
            "camera_extrinsics_c2w": c2w.astype(np.float32),
            "camera_extrinsics_w2c": w2c.astype(np.float32),
            "adjacent_relative_extrinsics_prev_to_cur": prev_to_cur.astype(np.float32),
            "adjacent_delta_yaw_pitch": adjacent_delta_yaw_pitch.astype(np.float32),
        }
    )
    return payload


def _extract_single_record_with_filter_extras(
    reconstructor: Any,
    device: Any,
    amp_dtype: Any,
    record: Any,
    input_mode: str,
    target_size: int,
    cam_token_mode: str,
    storage_dtype: str,
) -> dict[str, Any]:
    payload = BASE_EXTRACT_SINGLE_RECORD(
        reconstructor=reconstructor,
        device=device,
        amp_dtype=amp_dtype,
        record=record,
        input_mode=input_mode,
        target_size=target_size,
        cam_token_mode=cam_token_mode,
        storage_dtype=storage_dtype,
    )
    if isinstance(record, FilterMemoryRecord):
        payload = _augment_filter_payload(payload, record)
    return payload


def _validate_output_npz_with_filter_extras(
    output_path: str | Path,
    record: Any,
) -> tuple[bool, str]:
    ok, reason = BASE_VALIDATE_OUTPUT_NPZ(output_path, record)
    if not ok or not isinstance(record, FilterMemoryRecord):
        return ok, reason

    required_extra_keys = {
        "filtered_frame_ids",
        "observed_image_rel_paths",
        "prefix_frame_count",
        "sequence_dir_rel_path",
        "annotation_path",
        "source_sample_id",
        "camera_extrinsics_c2w",
        "camera_extrinsics_w2c",
        "adjacent_relative_extrinsics_prev_to_cur",
        "adjacent_delta_yaw_pitch",
    }
    try:
        with np.load(output_path, allow_pickle=False) as data:
            keys = set(data.files)
            missing_extra = sorted(required_extra_keys - keys)
            if missing_extra:
                return False, f"missing_filter_keys:{','.join(missing_extra)}"

            filtered_frame_ids = data["filtered_frame_ids"]
            observed_image_rel_paths = data["observed_image_rel_paths"]
            camera_extrinsics_c2w = data["camera_extrinsics_c2w"]
            camera_extrinsics_w2c = data["camera_extrinsics_w2c"]
            adjacent_relative = data["adjacent_relative_extrinsics_prev_to_cur"]
            adjacent_delta = data["adjacent_delta_yaw_pitch"]
            prefix_frame_count = int(LF._decode_scalar_from_npz(data["prefix_frame_count"]))

            num_views = len(record.image_rel_paths)
            if prefix_frame_count != int(record.prefix_frame_count):
                return False, "prefix_frame_count_mismatch"
            if filtered_frame_ids.shape[0] != num_views:
                return False, "filtered_frame_ids_shape_invalid"
            if observed_image_rel_paths.shape[0] != len(record.observed_image_rel_paths):
                return False, "observed_image_rel_paths_shape_invalid"
            if camera_extrinsics_c2w.shape != (num_views, 4, 4):
                return False, "camera_extrinsics_c2w_shape_invalid"
            if camera_extrinsics_w2c.shape != (num_views, 4, 4):
                return False, "camera_extrinsics_w2c_shape_invalid"
            if adjacent_relative.shape != (max(0, num_views - 1), 4, 4):
                return False, "adjacent_relative_shape_invalid"
            if adjacent_delta.shape != (max(0, num_views - 1), 2):
                return False, "adjacent_delta_shape_invalid"
    except Exception as exc:  # noqa: BLE001
        return False, f"load_filter_failed:{type(exc).__name__}"

    return True, "ok"


LF.extract_single_record = _extract_single_record_with_filter_extras
LF.validate_output_npz = _validate_output_npz_with_filter_extras
BASE.build_qa_latent_manifest = build_filter_memory_latent_manifest
BASE._load_sample_records_from_sequence_manifest_with_gt_pose = _load_filter_sample_records_from_sequence_manifest_with_gt_pose


def _validate_dataset_name(argv: list[str]) -> None:
    dataset_name = None
    for idx, token in enumerate(argv):
        if token == "--dataset_name" and idx + 1 < len(argv):
            dataset_name = argv[idx + 1].strip().lower()
            break
        if token.startswith("--dataset_name="):
            dataset_name = token.split("=", 1)[1].strip().lower()
            break
    if dataset_name is None:
        return
    if dataset_name not in FILTER_DATASET_NAMES:
        supported = ", ".join(sorted(FILTER_DATASET_NAMES))
        raise ValueError(
            f"this extractor only supports filter datasets. got={dataset_name}, supported={supported}"
        )


def main() -> None:
    _validate_dataset_name(sys.argv[1:])
    BASE.main()


if __name__ == "__main__":
    main()
