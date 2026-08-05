# 1. Tune hyperparams like seq_len, norm_eval, train_range, missing_image_replacement, large_image_sizes, feature_maps, datasets(xx1,x2,base)
_base_ = [
    "../default/vov_flash_480x640_baseline.py",
]

# The base config uses HTTPS `load_from` (VoV-99 init). That download can take a long time
# and shows 0% GPU until it finishes. Prefetch once, then point `load_from` at the file:
#   mkdir -p pretrained && wget -c -O pretrained/nuscenes_vov99_baseline_320x800.pth \
#     'https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth'
# load_from = "work_dirs/t4_base_vov_flash_480x640_bev_2_7_j6gen2/epoch_10.pth"
# load_from = "pretrained/best_NuScenesmetric_T4Metric_mAP_epoch_34.pth"
# load_from = "pretrained/nuscenes_vov99_baseline_320x800.pth"
load_from = "work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_gen2_base_partialignore_depth/epoch_34.pth"

# info_directory_path = "info/username/"
# data_root = "data/t4dataset/"
info_directory_path = "info/cameraonly/2_8/"
data_root = "data/"

data_prefix = _base_.data_prefix
class_names = _base_.class_names

batch_size = 8
num_workers = 16

num_epochs = 35
val_interval = 5

info_train_file_name="t4dataset_j6gen2_infos_train_with_visibility.pkl"
# info_val_file_name="t4dataset_j6gen2_infos_val.pkl"
info_val_file_name="t4dataset_j6gen2_infos_test.pkl"
info_test_file_name="t4dataset_j6gen2_infos_test.pkl"

camera_orders = {
    "J6_erga_Gen2": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"],
    "J6_x2_Gen2": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"],
    "JPNTaxi_xx1_Gen2": [
        "CAM_FRONT_WIDE",
        "CAM_FRONT_LEFT_WIDE",
        "CAM_BACK_LEFT_WIDE",
        "CAM_FRONT_RIGHT_WIDE",
        "CAM_BACK_RIGHT_WIDE",
    ],
    "JPNTaxi_solio_Gen2": [
        "CAM_FRONT_WIDE",
        "CAM_FRONT_LEFT_WIDE",
        "CAM_BACK_LEFT_WIDE",
        "CAM_FRONT_RIGHT_WIDE",
        "CAM_BACK_RIGHT_WIDE",
    ],
}

# `_base_` pulls multi-split `dataset_test_groups` from autoware_ml t4dataset/base.py.
# Without `_delete_=True`, MMEngine merges dicts and keeps j6gen2/base/... keys → missing pkls.
# `tools/detection3d/test.py` loops each group under `info_directory_path`.
dataset_test_groups = dict(
    _delete_=True,
    # base=(info_test_file_name, True),
    j6gen2=("t4dataset_j6gen2_infos_test.pkl", True),
)

# --- Training-time auxiliary depth supervision (does not affect inference) ---
# Insert LoadSparseDepthFromLiDAR AFTER ResizeCropFlipRotImage (2D aug baked into
# intrinsics) and BEFORE GlobalRotScaleTransImage (which virtually rotates/scales the
# LiDAR frame). Extend the train-only bundle collect_keys so the depth label is
# tensorized and stacked. Only the train pipeline / train dataset are touched; val/test
# stay unchanged so evaluation never tries to load point clouds.
sparse_depth_keys = ["sparse_depth", "sparse_depth_mask"]

# Normalized image-space polygons. x is horizontal, y is vertical.
# Used twice: Filter3DBoxesinBlindSpot paints these regions black in the images, and
# LoadSparseDepthFromLiDAR drops depth supervision on them (no point training the
# depth head to regress LiDAR returns from blacked-out ego-body pixels).
ego_vehicle_masks = dict(
    j6gen2=dict(
        CAM_FRONT_LEFT=[0.64, 1.0, 0.73, 0.65, 0.8, 0.65, 0.79, 0.74, 0.97, 0.83, 1.0, 0.73, 1.0, 1.0],
        CAM_FRONT_RIGHT=[0.19, 1.0, 0.21, 0.89, 0.38, 0.89, 0.39, 1.0],
        CAM_BACK_LEFT=[0.0, 0.0, 0.11, 0.0, 0.2, 1.0, 0.0, 1.0],
        CAM_BACK_RIGHT=[0.88, 0.0, 1.0, 0.0, 1.0, 1.0, 0.8, 1.0],
    ),
    largebus=dict(
        CAM_BACK_LEFT=[0.0, 0.45, 0.12, 0.48, 0.15, 1.0, 0.0, 1.0],
        CAM_BACK_RIGHT=[0.86, 0.49, 1.0, 0.4, 1.0, 1.0, 0.82, 1.0],
    ),
)

# Must match the AuxDepthHead depth_min/depth_max below. Points outside this range
# carry no usable bin label (beyond ~68m the fp32 Gaussian soft target underflows to
# all-zeros), so filter them at load time instead of diluting the loss mean.
depth_min = 1.0
depth_max = 61.2

train_pipeline = []
for _t in _base_.train_pipeline:
    _t = dict(_t)
    if _t["type"] == "PETRFormatBundle3D":
        _t["collect_keys"] = _base_.collect_keys + ["prev_exists"] + sparse_depth_keys
    train_pipeline.append(_t)
    if _t["type"] == "mmdet.ResizeCropFlipRotImage":
        # stride halved to match AuxDepthHead's upsample_factor=2 below
        train_pipeline.append(
            dict(
                type="LoadSparseDepthFromLiDAR",
                stride=_base_.stride // 2,
                load_dim=5,
                min_depth=depth_min,
                max_depth=depth_max,
            )
        )

train_collect_keys = _base_.collect_keys + ["img", "prev_exists", "img_metas"] + sparse_depth_keys

model = dict(
    use_aux_depth=True,
    aux_depth_head=dict(
        type="AuxDepthHead",
        in_channels=256,
        mid_channels=64,
        num_depth_bins=120,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_bin_sigma=1.0,  # Gaussian soft-target std, in bins
        upsample_factor=2,  # predict at 2x the neck's native stride
        loss_weight=0.1,  # keep below the 3D detection losses
    ),
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    dataset=dict(
        ann_file=info_directory_path + info_train_file_name,
        data_root=data_root,
        data_prefix=data_prefix,
        camera_orders=camera_orders,
        pipeline=train_pipeline,
        collect_keys=train_collect_keys,
        # NAS: StreamPETRDataset.filter_data() defaults to os.path.exists() per camera × every frame at init.
        # That dominates startup (num_workers cannot help). Skip when ann paths are trusted.
        # check_img_paths=False,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_val_file_name,
        data_root=data_root,
        data_prefix=data_prefix,
        camera_orders=camera_orders,
        # check_img_paths=False,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_test_file_name,
        data_root=data_root,
        data_prefix=data_prefix,
        camera_orders=camera_orders,
        # check_img_paths=False,
    ),
)


val_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_val_file_name)
test_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_test_file_name)


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)

lr = 5e-5
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)

# optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1),}))
optim_wrapper = dict(
    type="NoCacheAmpOptimWrapper",
    dtype="bfloat16",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    loss_scale="dynamic",
    clip_grad=dict(max_norm=1, norm_type=2),
)

# lrg policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-4,
    ),
]

auto_scale_lr = dict(base_batch_size=8, enable=True)

# model = dict(
#     pts_bbox_head=dict(
#         code_weights=[
#             3.0,
#             3.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#             1.0,
#         ],
#         assigner=dict(
#             reg_cost=dict(weight=0.5),
#         ),
#         loss_bbox=dict(loss_weight=0.5),
#     ),
# )

_train_pipeline_with_filters = []
for transform in train_pipeline:
    _train_pipeline_with_filters.append(transform)
    if transform.get("type") == "LoadAnnotations3D":
        _train_pipeline_with_filters.append(
            dict(
                type="Filter3DBoxesByVisibilityLevel",
                ignored_levels=("none",),
                debug_vis_dir="work_dirs/visibility_filter_vis_train",
                debug_vis_interval=10000,
                debug_vis_max_samples=10,
                debug_vis_kept=True,
                class_names=class_names,
            )
        )
        _train_pipeline_with_filters.append(
            dict(
                type="Filter3DBoxesinBlindSpot",
                visibility=0.03,
                min_visible_area=100,
                min_lidar_points=1,
                ego_masks=ego_vehicle_masks,
                mask_images=True,
                mask_color=(0, 0, 0),
                debug_vis_dir="work_dirs/blindspot_filter_vis_train",
                debug_vis_interval=10000,
                debug_vis_max_samples=10,
                debug_vis_kept=True,
                debug_log_interval=100,
                debug_log_max_samples=20,
                class_names=class_names,
            )
        )
train_pipeline = _train_pipeline_with_filters

train_dataloader["dataset"]["pipeline"] = train_pipeline

test_pipeline = []
for transform in _base_.test_pipeline:
    test_pipeline.append(transform)
    if transform.get("type") == "LoadAnnotations3D":
        test_pipeline.append(
            dict(
                type="Filter3DBoxesByVisibilityLevel",
                ignored_levels=("none",),
                debug_vis_dir="work_dirs/visibility_filter_vis_test",
                debug_vis_interval=10,
                debug_vis_max_samples=10,
                debug_vis_kept=True,
                class_names=class_names,
            )
        )

val_dataloader["dataset"]["pipeline"] = test_pipeline
test_dataloader["dataset"]["pipeline"] = test_pipeline
