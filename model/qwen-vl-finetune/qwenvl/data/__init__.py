import re
from pathlib import Path


# def _resolve_annotation_path(annotation_root, file_name):
#     # LlamaFactory can read shard directories, but qwenvl expects a single JSON/JSONL file.
#     if file_name == "QA_v4_w_thinking_shards":
#         file_name = "QA_v4_w_thinking.json"
#     return str(Path(annotation_root) / file_name)


def _dataset_config(
    *,
    annotation_root,
    file_name,
    data_path,
    provide_latent=False,
    latent_path=None,
    latent_field="latents",
    message_key="conversations",
    image_key="image",
    video_key="video",
    role_key="from",
    content_key="value",
    user_tag="human",
    assistant_tag="gpt",
):
    config = {
        "annotation_path": str(Path(annotation_root) / file_name), # _resolve_annotation_path(annotation_root, file_name),
        "data_path": data_path,
        "provide_latent": provide_latent,
        "message_key": message_key,
        "image_key": image_key,
        "video_key": video_key,
        "role_key": role_key,
        "content_key": content_key,
        "user_tag": user_tag,
        "assistant_tag": assistant_tag,
    }
    if latent_path is not None:
        config["latent_path"] = latent_path
        config["latent_field"] = latent_field
    return config


# Public placeholder datasets
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}


# ActivePerception / custom datasets synced from LlamaFactory/data/dataset_info.json
CA1M_REFERRING_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/CA1M_referring_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/CA1M_referring_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

CA1M_VACANT_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/CA1M_vacant_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/CA1M_vacant_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

PAP_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

PAP_512X384_FOV_90_LATENT = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_512x384_fov_90",
    file_name="QA_v6_w_latent_prompt.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_512x384_fov_90/images",
    provide_latent=True,
    latent_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data/PAP_512x384_fov_90",
    image_key="images",
    video_key="videos",
)

PAP_MULTI_STEP_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_multi_step_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_multi_step_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

PAP_FILTER_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_filter_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

PAP_FILTER_VQA_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_filter_VQA_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    # The VQA exporter reuses PAP_filter camera_image by default unless --materialize_images is enabled.
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

PAP_RETRIEVAL_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/PAP_retrieval_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/PAP_retrieval_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

RAW_PANO_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_512x384_fov_90_v2",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image",
    image_key="images",
    video_key="videos",
)

RAW_PANO_512X384_FOV_90_LATENT = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_512x384_fov_90_v2",
    file_name="QA_v6_w_latent_prompt.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_512x384_fov_90_v2/camera_image",
    provide_latent=True,
    latent_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data/raw_pano_512x384_fov_90",
    image_key="images",
    video_key="videos",
)

RAW_PANO_MULTI_STEP_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_multi_step_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_multi_step_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

RAW_PANO_FILTER_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_filter_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

RAW_PANO_FILTER_VQA_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_filter_VQA_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    # The VQA exporter reuses raw_pano_filter camera_image by default unless --materialize_images is enabled.
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

RAW_PANO_RETRIEVAL_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/raw_pano_retrieval_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/raw_pano_retrieval_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)


HSTAR_BENCH_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_bench_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_bench_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_512X384_FOV_90_LATENT = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_512x384_fov_90",
    file_name="QA_v6_w_latent_prompt.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_512x384_fov_90/images",
    provide_latent=True,
    latent_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data/hstar_sft_512x384_fov_90",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_OURS_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_512x384_fov_90/images",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_OURS_512X384_FOV_LATENT = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_512x384_fov_90",
    file_name="QA_v6_w_latent_prompt.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_512x384_fov_90/images",
    provide_latent=True,
    latent_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data/hstar_sft_ours_512x384_fov_90",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_OURS_FILTER_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_filter_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/hstar_sft_ours_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_OURS_FILTER_VQA_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_filter_VQA_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    # The VQA exporter reuses hstar_sft_ours_filter camera_image by default unless --materialize_images is enabled.
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/hstar_sft_ours_filter_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)

HSTAR_SFT_OURS_RETRIEVAL_512X384_FOV_90 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/vlm_data/hstar_sft_ours_retrieval_512x384_fov_90",
    file_name="QA_v4_w_thinking.json",
    data_path="/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/camera_data/hstar_sft_ours_retrieval_512x384_fov_90/camera_image",
    image_key="images",
    video_key="videos",
)


# LLAVA_1_5_LRV_MIX_965K = _dataset_config(
#     annotation_root="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/",
#     file_name="llava_v1_5_lrv_mix965k.json",
#     data_path="/share/project/emllm_mnt.1d/hpfs/baaiei/vlm/robobrain_train_images",
#     image_key="images",
#     video_key="videos",
# )


# llava_1_5_lrv_mix_965k = Dataset(
#     dataset_name="llava_1_5_lrv_mix_965k",
#     dataset_type="geometricdataset",
#     data_path="/share/project/zhouenshen/sfs/dataset/vlm/llava_v1_5_lrv_mix965k.json",
#     image_path="/share/project/emllm_mnt.1d/hpfs/baaiei/vlm/robobrain_train_images",
# )
# add_dataset(llava_1_5_lrv_mix_965k)

DEEPEYES_TRAIN_4k = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyes_train_4K",
    file_name="DeepEyes_train_4000.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyes_train_4K/data",
    image_key="images",
)

DEEPEYES_TRAIN_4k_ORG = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyes_train_4K",
    file_name="DeepEyes_train_4000_org.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyes_train_4K/data",
    image_key="images",
)

DEEPEYES_RL_82k = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava",
    file_name="DeepEyesV2_RL_82580.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava/images",
    image_key="images",
)

DEEPEYES_RL_82k_ORG = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava",
    file_name="DeepEyesV2_RL_82580_org.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava/images",
    image_key="images",
)

DEEPEYES_RL_82k_VSTAR = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava",
    file_name="DeepEyesV2_RL_82580_vstar.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava/images",
    image_key="images",
)

DEEPEYES_RL_82k_VSTAR_ORG = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava",
    file_name="DeepEyesV2_RL_82580_vstar_org.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/DeepEyesV2_RL_llava/images",
    image_key="images",
)

VISUALPROBE_TRAIN_5729 = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/VisualProbe_train",
    file_name="VisualProbe_train_5729.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/VisualProbe_train/data",
    image_key="images",
)

VISUALPROBE_TRAIN_5729_ORG = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/VisualProbe_train",
    file_name="VisualProbe_train_5729_org.json",
    data_path="/share/project/zhouenshen/sfs/dataset/vlm/VisualProbe_train/data",
    image_key="images",
)

LLAVA_ONEVISION_1_5_934k = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/vlm/",
    file_name="general_honey_data_rel_check_934694_tokenlt1350_metadata.json",
    data_path="/share/project/tanhuajie/robobrain_v2_5_sz/train_images",
    image_key="images",
)


VSI_590K = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/VSI/VSI-590K",
    file_name="metadata.json",
    data_path="/share/project/zhouenshen/sfs/dataset/VSI/VSI-590K",
    video_key="video",
)

LIBERO_FAST_VLM = _dataset_config(
    annotation_root="/share/project/zhouenshen/sfs/dataset/libero/libero_goal_fast_vlm",
    file_name="metadata.json",
    data_path="/share/project/zhouenshen/sfs/dataset/libero/libero_goal_fast_vlm",
    message_key="messages",
    image_key="images",
    role_key="role",
    content_key="content",
    user_tag="user",
    assistant_tag="assistant",
)


data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,

    "ca1m_referring_512x384_fov_90": CA1M_REFERRING_512X384_FOV_90,
    "ca1m_vacant_512x384_fov_90": CA1M_VACANT_512X384_FOV_90,

    "pap_512x384_fov_90": PAP_512X384_FOV_90,
    "pap_512x384_fov_90_latent": PAP_512X384_FOV_90_LATENT,
    "pap_multi_step_512x384_fov_90": PAP_MULTI_STEP_512X384_FOV_90,
    "pap_filter_512x384_fov_90": PAP_FILTER_512X384_FOV_90,
    "pap_filter_vqa_512x384_fov_90": PAP_FILTER_VQA_512X384_FOV_90,
    "pap_retrieval_512x384_fov_90": PAP_RETRIEVAL_512X384_FOV_90,

    "raw_pano_512x384_fov_90": RAW_PANO_512X384_FOV_90_LATENT,
    "raw_pano_512x384_fov_90_latent": RAW_PANO_512X384_FOV_90,
    "raw_pano_multi_step_512x384_fov_90": RAW_PANO_MULTI_STEP_512X384_FOV_90,
    "raw_pano_filter_512x384_fov_90": RAW_PANO_FILTER_512X384_FOV_90,
    "raw_pano_filter_vqa_512x384_fov_90": RAW_PANO_FILTER_VQA_512X384_FOV_90,
    "raw_pano_retrieval_512x384_fov_90": RAW_PANO_RETRIEVAL_512X384_FOV_90,

    "hstar_bench_512x384_fov_90": HSTAR_BENCH_512X384_FOV_90,
    "hstar_sft_512x384_fov_90": HSTAR_SFT_512X384_FOV_90,
    "hstar_sft_512x384_fov_90_latent": HSTAR_SFT_512X384_FOV_90_LATENT,

    "hstar_sft_ours_512x384_fov_90": HSTAR_SFT_OURS_512X384_FOV_90,
    "hstar_sft_ours_512x384_fov_90_latent": HSTAR_SFT_OURS_512X384_FOV_LATENT,
    "hstar_sft_ours_filter_512x384_fov_90": HSTAR_SFT_OURS_FILTER_512X384_FOV_90,
    "hstar_sft_ours_filter_vqa_512x384_fov_90": HSTAR_SFT_OURS_FILTER_VQA_512X384_FOV_90,
    "hstar_sft_ours_retrieval_512x384_fov_90": HSTAR_SFT_OURS_RETRIEVAL_512X384_FOV_90,

    "deepeyes_train": DEEPEYES_TRAIN_4k,
    "deepeyes_train_org": DEEPEYES_TRAIN_4k_ORG,
    "deepeyes_rl": DEEPEYES_RL_82k,
    "deepeyes_rl_org": DEEPEYES_RL_82k_ORG,
    "deepeyes_rl_vstar": DEEPEYES_RL_82k_VSTAR,
    "deepeyes_rl_vstar_org": DEEPEYES_RL_82k_VSTAR_ORG,
    "visualprobe_train": VISUALPROBE_TRAIN_5729,
    "visualprobe_train_org": VISUALPROBE_TRAIN_5729_ORG,
    "llava_onevision_1_5": LLAVA_ONEVISION_1_5_934k,
    "vsi_590k": VSI_590K,
    "libero_fast_vlm": LIBERO_FAST_VLM,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
