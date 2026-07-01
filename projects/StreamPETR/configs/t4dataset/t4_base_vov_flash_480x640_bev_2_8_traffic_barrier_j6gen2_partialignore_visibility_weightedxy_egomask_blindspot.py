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
load_from = "pretrained/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base/epoch_32.pth"

# info_directory_path = "info/username/"
# data_root = "data/t4dataset/"
# info_directory_path = "info/kokseang_2_8/"
info_directory_path = "info/cameraonly/kokseang_2_8/"
data_root = "data/"
class_names = _base_.class_names

batch_size = 4
num_workers = 32

num_epochs = 20
val_interval = 5

info_train_file_name="t4dataset_j6gen2_base_infos_train_with_visibility.pkl"
info_val_file_name="t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name="t4dataset_j6gen2_base_infos_test.pkl"

# `_base_` pulls multi-split `dataset_test_groups` from autoware_ml t4dataset/base.py.
# Without `_delete_=True`, MMEngine merges dicts and keeps j6gen2/base/... keys → missing pkls.
# `tools/detection3d/test.py` loops each group under `info_directory_path`.
dataset_test_groups = dict(
    _delete_=True,
    base=("t4dataset_base_infos_test.pkl", True),
    j6gen2=("t4dataset_j6gen2_infos_test.pkl", True),
    jpntaxi_gen2=("t4dataset_jpntaxi_gen2_infos_test.pkl", False),
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    dataset=dict(
        ann_file=info_directory_path + info_train_file_name,
        data_root=data_root,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_val_file_name,
        data_root=data_root,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_test_file_name,
        data_root=data_root,
    ),
)


val_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_val_file_name)
test_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_test_file_name)


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)

lr = 5e-6
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
    clip_grad=dict(max_norm=1.0, norm_type=2),
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

auto_scale_lr = dict(base_batch_size=8, enable=False)


# Normalized image-space polygons. x is horizontal, y is vertical.
ego_vehicle_masks = dict(
    j6gen2=dict(
        CAM_FRONT_LEFT=[0.64, 1.0, 0.73, 0.65, 0.8, 0.65, 0.79, 0.74, 0.97, 0.83, 1.0, 0.73, 1.0, 1.0],
        CAM_FRONT_RIGHT=[0.19, 1.0, 0.21, 0.89, 0.38, 0.89, 0.39, 1.0],
        CAM_BACK_LEFT=[0.88, 0.0, 1.0, 0.0, 1.0, 1.0, 0.8, 1.0],
        CAM_BACK_RIGHT=[0.0, 0.0, 0.11, 0.0, 0.2, 1.0, 0.0, 1.0],
    ),
    largebus=dict(
        CAM_BACK_LEFT=[0.0, 0.45, 0.12, 0.48, 0.15, 1.0, 0.0, 1.0],
        CAM_BACK_RIGHT=[0.85, 0.48, 1.0, 0.4, 1.0, 1.0, 0.81, 1.0],
    ),
)

model = dict(
    pts_bbox_head=dict(
        code_weights=[
            3.0,
            3.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        assigner=dict(
            reg_cost=dict(weight=0.5),
        ),
        loss_bbox=dict(loss_weight=0.5),
    ),
)

train_pipeline = []
for transform in _base_.train_pipeline:
    train_pipeline.append(transform)
    if transform.get("type") == "LoadAnnotations3D":
        train_pipeline.append(
            dict(
                type="Filter3DBoxesByVisibilityLevel",
                ignored_levels=("none",),
                debug_vis_dir="work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_visibility/visibility_filter_vis_train",
                debug_vis_interval=10000,
                debug_vis_max_samples=10,
                debug_vis_kept=True,
                class_names=class_names,
            )
        )
        train_pipeline.append(
            dict(
                type="Filter3DBoxesinBlindSpot",
                visibility=0.05,
                min_visible_area=800,
                min_lidar_points=3,
                ego_masks=ego_vehicle_masks,
                mask_images=True,
                mask_color=(0, 0, 0),
                debug_vis_dir="work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_visibility_weightedxy_egomask_blindspot/blindspot_filter_vis_train",
                debug_vis_interval=10000,
                debug_vis_max_samples=10,
                debug_vis_kept=True,
                debug_log_interval=1000,
                debug_log_max_samples=200,
                class_names=class_names,
            )
        )
        # train_pipeline.append(
        #     dict(
        #         type="Filter3DBoxesinBlindSpot",
        #         visibility=0.05,
        #         min_visible_area=800,
        #         min_lidar_points=3,
        #         # debug_vis_dir="work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_blindfiltered/blindspot_filter_vis",
        #         debug_vis_dir=None,
        #         debug_vis_interval=1,
        #         debug_vis_max_samples=100,
        #         debug_vis_kept=True,
        #         debug_log_interval=1,
        #         debug_log_max_samples=0,
        #         class_names=class_names,
        #     )
        # )

train_dataloader["dataset"]["pipeline"] = train_pipeline

test_pipeline = []
for transform in _base_.test_pipeline:
    test_pipeline.append(transform)
    if transform.get("type") == "LoadAnnotations3D":
        test_pipeline.append(
            dict(
                type="Filter3DBoxesByVisibilityLevel",
                ignored_levels=("none",),
                debug_vis_dir="work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_visibility/visibility_filter_vis_test",
                debug_vis_interval=1,
                debug_vis_max_samples=100,
                debug_vis_kept=True,
                class_names=class_names,
            )
        )

val_dataloader["dataset"]["pipeline"] = test_pipeline
test_dataloader["dataset"]["pipeline"] = test_pipeline
