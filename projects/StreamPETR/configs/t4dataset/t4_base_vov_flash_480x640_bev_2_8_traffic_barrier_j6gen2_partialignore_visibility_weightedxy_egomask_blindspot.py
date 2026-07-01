_base_ = [
    "./t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_visibility_weightedxy.py",
]

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

train_pipeline = []
for transform in _base_.train_pipeline:
    train_pipeline.append(transform)
    if transform.get("type") == "Filter3DBoxesByVisibilityLevel":
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

train_dataloader["dataset"]["pipeline"] = train_pipeline
