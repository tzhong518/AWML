# Copyright (c) OpenMMLab. All rights reserved.
"""Test StreamPETR once and report metrics for different BEV angle sectors.

The model input stays unchanged, e.g. the configured 5 cameras. This script
only filters predictions and GTs by BEV angle before evaluation so we can check
whether missing CAM_BACK hurts rear-sector accuracy.
"""

import argparse
import copy
import os
import os.path as osp
import pickle
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.utils import replace_ceph_backend

from autoware_ml.detection3d.evaluation.t4metric.t4metric import T4Metric


# Angles are degrees in ego/LiDAR BEV, computed as atan2(y, x).
# Assumption: x is forward and y is left, which matches common LiDAR box coords.
ANGLE_SECTORS = {
    "all": [(-180.0, 180.0)],
    "front": [(-45.0, 45.0)],
    "front_left": [(0.0, 90.0)],
    "left": [(45.0, 135.0)],
    "side_left": [(60.0, 120.0)],
    "rear_left": [(105.0, 150.0)],
    "rear_center": [(150.0, 180.0), (-180.0, -150.0)],
    "rear": [(135.0, 180.0), (-180.0, -135.0)],
    "rear_right": [(-150.0, -105.0)],
    "side_right": [(-120.0, -60.0)],
    "right": [(-135.0, -45.0)],
    "front_right": [(-90.0, 0.0)],
}


def _parse_angle_ranges(spec: str) -> Tuple[str, List[Tuple[float, float]]]:
    """Parse NAME=START:END[,START:END] into angle ranges."""
    if "=" not in spec:
        raise ValueError(f"Invalid sector {spec!r}. Expected NAME=START:END[,START:END].")
    name, range_spec = spec.split("=", 1)
    name = name.strip()
    ranges = []
    for item in range_spec.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid range {item!r} in {spec!r}. Expected START:END.")
        start, end = item.split(":", 1)
        ranges.append((float(start), float(end)))
    if not name or not ranges:
        raise ValueError(f"Invalid sector {spec!r}. Expected NAME=START:END[,START:END].")
    return name, ranges


def _angle_mask_xy(xy: torch.Tensor, ranges: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Return mask for centers whose atan2(y, x) falls in any range."""
    if xy.numel() == 0:
        return torch.zeros((xy.shape[0],), dtype=torch.bool, device=xy.device)

    angles = torch.rad2deg(torch.atan2(xy[:, 1], xy[:, 0]))
    mask = torch.zeros_like(angles, dtype=torch.bool)
    for start, end in ranges:
        if start <= end:
            mask |= (angles >= start) & (angles < end)
        else:
            # Wrap-around range, e.g. 135:-135.
            mask |= (angles >= start) | (angles < end)
    return mask


def _filter_lidar_boxes_by_angle(bboxes_3d, ranges: Sequence[Tuple[float, float]]):
    if not isinstance(bboxes_3d, LiDARInstance3DBoxes):
        return bboxes_3d, None
    centers = bboxes_3d.tensor[:, :2]
    mask = _angle_mask_xy(centers, ranges)
    return bboxes_3d[mask], mask


def _filter_array_like(value, mask):
    if mask is None:
        return value
    if value is None:
        return value
    mask_np = mask.detach().cpu().numpy()
    if torch.is_tensor(value):
        if value.shape[:1] == (mask.numel(),):
            return value[mask.to(value.device)]
        if value.numel() == mask.numel():
            return value.reshape(-1)[mask.to(value.device)]
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == mask.numel():
            return type(value)(item for item, keep in zip(value, mask_np) if keep)
        if len(value) == 1:
            return _filter_array_like(value[0], mask)
    arr = np.asarray(value)
    if arr.shape[:1] == (mask.numel(),):
        return arr[mask_np]
    if arr.size == mask.numel():
        return arr.reshape(-1)[mask_np]
    return value


def _get_field(container, key):
    if isinstance(container, dict):
        return container[key]
    return getattr(container, key)


def _has_field(container, key):
    if isinstance(container, dict):
        return key in container
    return hasattr(container, key)


def _set_field(container, key, value):
    if isinstance(container, dict):
        container[key] = value
    else:
        setattr(container, key, value)


def _copy_and_filter_pred(pred_3d, ranges: Sequence[Tuple[float, float]]):
    pred = copy.deepcopy(pred_3d)
    filtered_boxes, mask = _filter_lidar_boxes_by_angle(_get_field(pred, "bboxes_3d"), ranges)
    _set_field(pred, "bboxes_3d", filtered_boxes)
    if _has_field(pred, "scores_3d"):
        _set_field(pred, "scores_3d", _filter_array_like(_get_field(pred, "scores_3d"), mask))
    if _has_field(pred, "labels_3d"):
        _set_field(pred, "labels_3d", _filter_array_like(_get_field(pred, "labels_3d"), mask))
    return pred


def _copy_and_filter_gt(gt_3d: Dict, ranges: Sequence[Tuple[float, float]]):
    gt = copy.copy(gt_3d)
    filtered_boxes, mask = _filter_lidar_boxes_by_angle(gt["bboxes_3d"], ranges)
    gt["bboxes_3d"] = filtered_boxes
    for key in ("scores_3d", "labels_3d", "num_lidar_pts"):
        if key in gt:
            gt[key] = _filter_array_like(gt[key], mask)
    return gt


def _copy_and_filter_pred_by_mask(pred_3d, keep_mask: torch.Tensor):
    pred = copy.deepcopy(pred_3d)
    _set_field(pred, "bboxes_3d", _get_field(pred, "bboxes_3d")[keep_mask])
    if _has_field(pred, "scores_3d"):
        _set_field(pred, "scores_3d", _filter_array_like(_get_field(pred, "scores_3d"), keep_mask))
    if _has_field(pred, "labels_3d"):
        _set_field(pred, "labels_3d", _filter_array_like(_get_field(pred, "labels_3d"), keep_mask))
    return pred


def _copy_and_filter_gt_by_mask(gt_3d: Dict, keep_mask: torch.Tensor):
    gt = copy.copy(gt_3d)
    gt["bboxes_3d"] = gt["bboxes_3d"][keep_mask]
    for key in ("scores_3d", "labels_3d", "num_lidar_pts"):
        if key in gt:
            gt[key] = _filter_array_like(gt[key], keep_mask)
    return gt


def _filter_results_by_angle(results: List[dict], ranges: Sequence[Tuple[float, float]]) -> List[dict]:
    filtered = []
    for result in results:
        item = copy.copy(result)
        item["pred_instances_3d"] = _copy_and_filter_pred(result["pred_instances_3d"], ranges)
        item["gt_instances_3d"] = _copy_and_filter_gt(result["gt_instances_3d"], ranges)
        filtered.append(item)
    return filtered


def _ego_rear_strip_mask(
    bboxes_3d,
    ego_width: float,
    rear_distance: float,
    use_entire_box: bool = True,
) -> torch.Tensor:
    if not isinstance(bboxes_3d, LiDARInstance3DBoxes):
        return torch.zeros((0,), dtype=torch.bool)
    if len(bboxes_3d) == 0:
        return torch.zeros((0,), dtype=torch.bool, device=bboxes_3d.tensor.device)

    half_width = ego_width / 2.0
    if use_entire_box:
        corners = bboxes_3d.corners[:, :, :2]
        x = corners[:, :, 0]
        y = corners[:, :, 1]
        return (x >= -rear_distance).all(dim=1) & (x < 0.0).all(dim=1) & (y.abs() <= half_width).all(dim=1)

    centers = bboxes_3d.tensor[:, :2]
    return (centers[:, 0] >= -rear_distance) & (centers[:, 0] < 0.0) & (centers[:, 1].abs() <= half_width)


def _apply_ego_rear_score_suppression_to_pred(
    pred_3d,
    ego_width: float,
    rear_distance: float,
    score_scale: float,
    use_entire_box: bool,
):
    pred = copy.deepcopy(pred_3d)
    if not _has_field(pred, "scores_3d"):
        return pred, 0

    bboxes_3d = _get_field(pred, "bboxes_3d")
    mask = _ego_rear_strip_mask(bboxes_3d, ego_width, rear_distance, use_entire_box)
    suppressed_count = int(mask.sum().item()) if torch.is_tensor(mask) else 0
    if suppressed_count == 0:
        return pred, 0

    scores = _get_field(pred, "scores_3d")
    if torch.is_tensor(scores):
        new_scores = scores.clone()
        new_scores[mask.to(scores.device)] = new_scores[mask.to(scores.device)] * score_scale
    else:
        mask_np = mask.detach().cpu().numpy()
        new_scores = np.asarray(scores).copy()
        new_scores.reshape(-1)[mask_np] = new_scores.reshape(-1)[mask_np] * score_scale
    _set_field(pred, "scores_3d", new_scores)
    return pred, suppressed_count


def _apply_ego_rear_score_suppression(
    results: List[dict],
    ego_width: float,
    rear_distance: float,
    score_scale: float,
    use_entire_box: bool,
) -> Tuple[List[dict], int]:
    if score_scale >= 1.0:
        return results, 0

    suppressed_results = []
    total_suppressed = 0
    for result in results:
        item = copy.copy(result)
        item["pred_instances_3d"], suppressed_count = _apply_ego_rear_score_suppression_to_pred(
            result["pred_instances_3d"],
            ego_width,
            rear_distance,
            score_scale,
            use_entire_box,
        )
        total_suppressed += suppressed_count
        suppressed_results.append(item)
    return suppressed_results, total_suppressed


def _exclude_ego_rear_strip(
    results: List[dict],
    ego_width: float,
    rear_distance: float,
    use_entire_box: bool,
) -> Tuple[List[dict], int, int]:
    filtered_results = []
    removed_pred = 0
    removed_gt = 0
    for result in results:
        pred_boxes = _get_field(result["pred_instances_3d"], "bboxes_3d")
        gt_boxes = result["gt_instances_3d"]["bboxes_3d"]

        pred_drop = _ego_rear_strip_mask(pred_boxes, ego_width, rear_distance, use_entire_box)
        gt_drop = _ego_rear_strip_mask(gt_boxes, ego_width, rear_distance, use_entire_box)
        pred_keep = ~pred_drop
        gt_keep = ~gt_drop

        item = copy.copy(result)
        item["pred_instances_3d"] = _copy_and_filter_pred_by_mask(result["pred_instances_3d"], pred_keep)
        item["gt_instances_3d"] = _copy_and_filter_gt_by_mask(result["gt_instances_3d"], gt_keep)
        removed_pred += int(pred_drop.sum().item())
        removed_gt += int(gt_drop.sum().item())
        filtered_results.append(item)
    return filtered_results, removed_pred, removed_gt


def _count_boxes(results: List[dict]) -> Tuple[int, int]:
    pred_count = 0
    gt_count = 0
    for result in results:
        pred_count += len(_get_field(result["pred_instances_3d"], "bboxes_3d"))
        gt_count += len(result["gt_instances_3d"]["bboxes_3d"])
    return pred_count, gt_count


def _as_numpy_1d(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _labels_to_names(labels, class_names: Sequence[str]) -> List[str]:
    names = []
    for label in _as_numpy_1d(labels):
        label_idx = int(label)
        if 0 <= label_idx < len(class_names):
            names.append(class_names[label_idx])
        else:
            names.append(str(label_idx))
    return names


def _count_by_name(names: Iterable[str]) -> Dict[str, int]:
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _distance_bucket_counts(bboxes_3d, buckets: Sequence[Tuple[float, float]]) -> Dict[str, int]:
    counts = {f"{start:g}-{end:g}m": 0 for start, end in buckets}
    if not isinstance(bboxes_3d, LiDARInstance3DBoxes) or len(bboxes_3d) == 0:
        return counts
    distances = torch.linalg.norm(bboxes_3d.tensor[:, :2], dim=1).detach().cpu().numpy()
    for start, end in buckets:
        counts[f"{start:g}-{end:g}m"] = int(((distances >= start) & (distances < end)).sum())
    return counts


def _summarize_results(results: List[dict], class_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    pred_names = []
    gt_names = []
    pred_distance_counts = {"0-20m": 0, "20-40m": 0, "40-60m": 0}
    gt_distance_counts = {"0-20m": 0, "20-40m": 0, "40-60m": 0}
    buckets = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]

    for result in results:
        pred = result["pred_instances_3d"]
        gt = result["gt_instances_3d"]
        pred_names.extend(_labels_to_names(_get_field(pred, "labels_3d"), class_names))
        gt_names.extend(_labels_to_names(gt["labels_3d"], class_names))
        for key, value in _distance_bucket_counts(_get_field(pred, "bboxes_3d"), buckets).items():
            pred_distance_counts[key] += value
        for key, value in _distance_bucket_counts(gt["bboxes_3d"], buckets).items():
            gt_distance_counts[key] += value

    return {
        "pred_class": _count_by_name(pred_names),
        "gt_class": _count_by_name(gt_names),
        "pred_distance": pred_distance_counts,
        "gt_distance": gt_distance_counts,
    }


def _format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))


def _boxes_xy_numpy(bboxes_3d) -> np.ndarray:
    if not isinstance(bboxes_3d, LiDARInstance3DBoxes) or len(bboxes_3d) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return bboxes_3d.tensor[:, :2].detach().cpu().numpy()


def _distance_mask_xy(xy: np.ndarray, distance_range: Optional[Tuple[float, float]]) -> np.ndarray:
    if distance_range is None:
        return np.ones((xy.shape[0],), dtype=bool)
    start, end = distance_range
    distances = np.linalg.norm(xy, axis=1)
    return (distances >= start) & (distances < end)


def _compute_operating_point(
    results: List[dict],
    class_idx: int,
    score_thr: float,
    dist_thr: float,
    distance_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Compute greedy same-class center-distance precision/recall."""
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0

    for result in results:
        pred = result["pred_instances_3d"]
        gt = result["gt_instances_3d"]

        pred_xy = _boxes_xy_numpy(_get_field(pred, "bboxes_3d"))
        gt_xy = _boxes_xy_numpy(gt["bboxes_3d"])
        pred_labels = _as_numpy_1d(_get_field(pred, "labels_3d")).astype(np.int64)
        gt_labels = _as_numpy_1d(gt["labels_3d"]).astype(np.int64)
        pred_scores = _as_numpy_1d(_get_field(pred, "scores_3d")).astype(np.float32)

        pred_mask = (pred_labels == class_idx) & (pred_scores >= score_thr)
        pred_mask &= _distance_mask_xy(pred_xy, distance_range)
        gt_mask = gt_labels == class_idx
        gt_mask &= _distance_mask_xy(gt_xy, distance_range)

        frame_pred_xy = pred_xy[pred_mask]
        frame_pred_scores = pred_scores[pred_mask]
        frame_gt_xy = gt_xy[gt_mask]

        total_gt += len(frame_gt_xy)
        total_pred += len(frame_pred_xy)
        if len(frame_pred_xy) == 0:
            continue

        matched_gt = np.zeros((len(frame_gt_xy),), dtype=bool)
        order = np.argsort(-frame_pred_scores)
        for pred_idx in order:
            if len(frame_gt_xy) == 0:
                total_fp += 1
                continue
            distances = np.linalg.norm(frame_gt_xy - frame_pred_xy[pred_idx][None, :], axis=1)
            nearest_gt_idx = int(np.argmin(distances))
            if distances[nearest_gt_idx] <= dist_thr and not matched_gt[nearest_gt_idx]:
                matched_gt[nearest_gt_idx] = True
                total_tp += 1
            else:
                total_fp += 1

    total_fn = total_gt - total_tp
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    return {
        "gt": total_gt,
        "pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
    }


def _format_distance_range(distance_range: Optional[Tuple[float, float]]) -> str:
    if distance_range is None:
        return "all"
    return f"{distance_range[0]:g}-{distance_range[1]:g}m"


def _log_operating_summary(
    sector_name: str,
    results: List[dict],
    class_names: Sequence[str],
    summary_classes: Sequence[str],
    score_thrs: Sequence[float],
    dist_thrs: Sequence[float],
    logger,
) -> None:
    if logger is None:
        logger = MMLogger.get_current_instance()
    distance_ranges = [None, (0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    print_log(f"[{sector_name}] operating-point summary", logger=logger)
    print_log(
        "| class | range | score_thr | dist_thr | GT | Pred | TP | FP | FN | precision | recall |",
        logger=logger,
    )
    print_log("| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |", logger=logger)
    for class_name in summary_classes:
        if class_name not in class_names:
            print_log(f"[{sector_name}] skip unknown summary class: {class_name}", logger=logger)
            continue
        class_idx = class_names.index(class_name)
        for distance_range in distance_ranges:
            range_name = _format_distance_range(distance_range)
            for score_thr in score_thrs:
                for dist_thr in dist_thrs:
                    row = _compute_operating_point(results, class_idx, score_thr, dist_thr, distance_range)
                    print_log(
                        f"| {class_name} | {range_name} | {score_thr:.3f} | {dist_thr:.2f} | "
                        f"{row['gt']} | {row['pred']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                        f"{row['precision']:.4f} | {row['recall']:.4f} |",
                        logger=logger,
                    )


def _log_sector_summary(sector_name: str, results: List[dict], class_names: Sequence[str], logger) -> None:
    if logger is None:
        logger = MMLogger.get_current_instance()
    pred_count, gt_count = _count_boxes(results)
    summary = _summarize_results(results, class_names)
    print_log(f"[{sector_name}] pred boxes: {pred_count}, gt boxes: {gt_count}", logger=logger)
    print_log(f"[{sector_name}] pred class counts: {_format_counts(summary['pred_class'])}", logger=logger)
    print_log(f"[{sector_name}] gt class counts: {_format_counts(summary['gt_class'])}", logger=logger)
    print_log(f"[{sector_name}] pred distance counts: {_format_counts(summary['pred_distance'])}", logger=logger)
    print_log(f"[{sector_name}] gt distance counts: {_format_counts(summary['gt_distance'])}", logger=logger)


@METRICS.register_module()
class T4AngleMetric(T4Metric):
    """T4Metric wrapper that reports metrics per BEV angle sector."""

    def __init__(
        self,
        angle_sectors: Dict[str, Sequence[Tuple[float, float]]] = None,
        include_overall: bool = True,
        results_pkl: str = None,
        summary_classes: Sequence[str] = ("pedestrian",),
        summary_score_thrs: Sequence[float] = (0.1,),
        summary_dist_thrs: Sequence[float] = (0.5, 1.0, 2.0),
        ego_rear_score_scale: float = 1.0,
        ego_rear_width: float = 2.4,
        ego_rear_distance: float = 60.0,
        ego_rear_mode: str = "entire_box",
        exclude_ego_rear_strip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.angle_sectors = angle_sectors or {"rear": ANGLE_SECTORS["rear"]}
        self.include_overall = include_overall
        self.results_pkl = results_pkl
        self.summary_classes = list(summary_classes)
        self.summary_score_thrs = list(summary_score_thrs)
        self.summary_dist_thrs = list(summary_dist_thrs)
        self.ego_rear_score_scale = ego_rear_score_scale
        self.ego_rear_width = ego_rear_width
        self.ego_rear_distance = ego_rear_distance
        self.ego_rear_mode = ego_rear_mode
        self.exclude_ego_rear_strip = exclude_ego_rear_strip

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        metric_dict = {}
        logger = MMLogger.get_current_instance()
        original_jsonfile_prefix = self.jsonfile_prefix

        if self.results_pkl:
            os.makedirs(osp.dirname(osp.abspath(self.results_pkl)), exist_ok=True)
            with open(self.results_pkl, "wb") as f:
                pickle.dump(results, f)
            print_log(f"Saved processed T4AngleMetric results to: {self.results_pkl}", logger=logger)

        if self.exclude_ego_rear_strip:
            results, removed_pred, removed_gt = _exclude_ego_rear_strip(
                results,
                ego_width=self.ego_rear_width,
                rear_distance=self.ego_rear_distance,
                use_entire_box=self.ego_rear_mode == "entire_box",
            )
            print_log(
                "Excluded ego rear strip from evaluation: "
                f"width={self.ego_rear_width:g}m, distance={self.ego_rear_distance:g}m, "
                f"mode={self.ego_rear_mode}, removed_pred_boxes={removed_pred}, removed_gt_boxes={removed_gt}",
                logger=logger,
            )

        if self.ego_rear_score_scale < 1.0:
            results, suppressed_count = _apply_ego_rear_score_suppression(
                results,
                ego_width=self.ego_rear_width,
                rear_distance=self.ego_rear_distance,
                score_scale=self.ego_rear_score_scale,
                use_entire_box=self.ego_rear_mode == "entire_box",
            )
            print_log(
                "Applied ego rear score suppression: "
                f"scale={self.ego_rear_score_scale:g}, width={self.ego_rear_width:g}m, "
                f"distance={self.ego_rear_distance:g}m, mode={self.ego_rear_mode}, "
                f"suppressed_pred_boxes={suppressed_count}",
                logger=logger,
            )

        if self.include_overall:
            if original_jsonfile_prefix:
                self.jsonfile_prefix = osp.join(original_jsonfile_prefix, "overall")
            _log_sector_summary("overall", results, self.class_names, logger)
            _log_operating_summary(
                "overall",
                results,
                self.class_names,
                self.summary_classes,
                self.summary_score_thrs,
                self.summary_dist_thrs,
                logger,
            )
            overall_metrics = super().compute_metrics(results)
            metric_dict.update({f"overall/{key}": value for key, value in overall_metrics.items()})

        for sector_name, ranges in self.angle_sectors.items():
            sector_results = _filter_results_by_angle(results, ranges)
            if original_jsonfile_prefix:
                self.jsonfile_prefix = osp.join(original_jsonfile_prefix, sector_name)
            _log_sector_summary(sector_name, sector_results, self.class_names, logger)
            _log_operating_summary(
                sector_name,
                sector_results,
                self.class_names,
                self.summary_classes,
                self.summary_score_thrs,
                self.summary_dist_thrs,
                logger,
            )
            sector_metrics = super().compute_metrics(sector_results)
            metric_dict.update({f"{sector_name}/{key}": value for key, value in sector_metrics.items()})

        self.jsonfile_prefix = original_jsonfile_prefix
        return metric_dict


def _iter_metrics(evaluator) -> Iterable[BaseMetric]:
    if hasattr(evaluator, "metrics"):
        yield from evaluator.metrics
    else:
        yield evaluator


def _find_angle_metric(evaluator) -> T4AngleMetric:
    for metric in _iter_metrics(evaluator):
        if isinstance(metric, T4AngleMetric):
            return metric
    raise RuntimeError("T4AngleMetric was not found in runner.test_evaluator.")


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test StreamPETR and evaluate BEV angle sectors")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="the directory to save the file containing evaluation metrics")
    parser.add_argument("--out", help="the pkl file to dump raw test results.")
    parser.add_argument(
        "--results-pkl",
        help=(
            "Path to save/load T4AngleMetric internal results. If omitted, an automatic "
            "path under work_dir is used. If the file exists, the script skips model "
            "inference and recomputes metrics from this pickle."
        ),
    )
    parser.add_argument("--ceph", action="store_true", help="Use ceph as data storage backend")
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument("--score-thr", type=float, default=0.1, help="bbox score threshold")
    parser.add_argument(
        "--task",
        type=str,
        choices=["mono_det", "multi-view_det", "lidar_det", "lidar_seg", "multi-modality_det"],
        help="Determine the visualization method depending on the task.",
    )
    parser.add_argument("--wait-time", type=float, default=2, help="the interval of show (s)")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    parser.add_argument(
        "--angle-sectors",
        nargs="+",
        default=[],#["rear", "front", "left", "right"],
        help=(
            "Built-in angle sectors to evaluate. Available: "
            + ", ".join(sorted(ANGLE_SECTORS))
            + ". Use --custom-angle-sector for custom ranges."
        ),
    )
    parser.add_argument(
        "--custom-angle-sector",
        action="append",
        default=[],
        metavar="NAME=START:END[,START:END]",
        help="Add custom BEV angle sector in degrees, e.g. rear_center=150:180,-180:-150.",
    )
    parser.add_argument(
        "--summary-classes",
        nargs="+",
        default=["pedestrian"],
        help="Classes for operating-point precision/recall summary.",
    )
    parser.add_argument(
        "--summary-score-thrs",
        nargs="+",
        type=float,
        default=[0.1],
        help="Score thresholds for operating-point precision/recall summary.",
    )
    parser.add_argument(
        "--summary-dist-thrs",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="Center-distance match thresholds in meters for operating-point precision/recall summary.",
    )
    parser.add_argument(
        "--ego-rear-score-scale",
        type=float,
        default=1.0,
        help="Multiply scores of predictions in ego rear strip by this value. 1 disables suppression.",
    )
    parser.add_argument(
        "--ego-rear-width",
        type=float,
        default=2.4,
        help="Ego rear strip width in meters.",
    )
    parser.add_argument(
        "--ego-rear-distance",
        type=float,
        default=60.0,
        help="Ego rear strip length behind ego in meters.",
    )
    parser.add_argument(
        "--ego-rear-mode",
        choices=("entire_box", "center"),
        default="entire_box",
        help="Whether suppression uses the entire box footprint or only box center.",
    )
    parser.add_argument(
        "--exclude-ego-rear-strip",
        action="store_true",
        help=(
            "Exclude predictions and GTs in the ego rear strip from evaluation. "
            "The strip is controlled by --ego-rear-width, --ego-rear-distance, and --ego-rear-mode."
        ),
    )
    parser.add_argument("--no-overall", action="store_true", help="Do not report unfiltered overall metrics.")
    # When using PyTorch version >= 2.0.0, torch.distributed.launch passes
    # --local-rank instead of --local_rank.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def add_dump_results_metric(runner, out_path):
    if out_path is None:
        return
    assert out_path.endswith((".pkl", ".pickle")), "The dump file must be a pkl file."
    runner.test_evaluator.metrics.append(DumpResults(out_file_path=out_path))


def get_group_output_path(out_path, dataset_name):
    if out_path is None:
        return None
    root, ext = osp.splitext(out_path)
    if root.endswith(f"_{dataset_name}"):
        return out_path
    return f"{root}_{dataset_name}{ext}"


def get_default_results_pkl(work_dir, checkpoint):
    checkpoint_stem = osp.splitext(osp.basename(checkpoint))[0]
    checkpoint_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_stem)
    return osp.join(work_dir, f"t4angle_processed_results_{checkpoint_stem}.pkl")


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if "visualization" in default_hooks:
        visualization_hook = default_hooks["visualization"]
        visualization_hook["draw"] = True
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.show_dir:
            visualization_hook["test_out_dir"] = args.show_dir
        all_task_choices = ["mono_det", "multi-view_det", "lidar_det", "lidar_seg", "multi-modality_det"]
        assert args.task in all_task_choices, (
            "You must set " f"'--task' in {all_task_choices} in the command " "if you want to use visualization hook"
        )
        visualization_hook["vis_task"] = args.task
        visualization_hook["score_thr"] = args.score_thr
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    return cfg


def resolve_angle_sectors(args) -> Dict[str, Sequence[Tuple[float, float]]]:
    sectors = {}
    custom = dict(_parse_angle_ranges(spec) for spec in args.custom_angle_sector)
    available = {**ANGLE_SECTORS, **custom}
    for sector_name in args.angle_sectors:
        if sector_name not in available:
            raise KeyError(f"Unknown angle sector {sector_name!r}. Available: {', '.join(sorted(available))}")
        sectors[sector_name] = available[sector_name]
    return sectors


def install_angle_metric(cfg, angle_sectors, include_overall, args):
    cfg.test_evaluator.type = "T4AngleMetric"
    cfg.test_evaluator.angle_sectors = angle_sectors
    cfg.test_evaluator.include_overall = include_overall
    cfg.test_evaluator.summary_classes = args.summary_classes
    cfg.test_evaluator.summary_score_thrs = args.summary_score_thrs
    cfg.test_evaluator.summary_dist_thrs = args.summary_dist_thrs
    cfg.test_evaluator.ego_rear_score_scale = args.ego_rear_score_scale
    cfg.test_evaluator.ego_rear_width = args.ego_rear_width
    cfg.test_evaluator.ego_rear_distance = args.ego_rear_distance
    cfg.test_evaluator.ego_rear_mode = args.ego_rear_mode
    cfg.test_evaluator.exclude_ego_rear_strip = args.exclude_ego_rear_strip
    return cfg


def build_runner(cfg):
    if "runner_type" not in cfg:
        return Runner.from_cfg(cfg)
    return RUNNERS.build(cfg)


def evaluate_from_results_pkl(cfg, results_pkl, log_message):
    runner = build_runner(cfg)
    metric = _find_angle_metric(runner.test_evaluator)
    metric.logger = runner.logger
    print_log(f"Using processed results pickle: {results_pkl}", logger=runner.logger)
    print_log(f"{log_message}. Loading processed results from: {results_pkl}", logger=runner.logger)
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    metrics = metric.compute_metrics(results)
    runner.logger.info(metrics)
    return runner


def test_and_optionally_save_results(cfg, out_path, log_message):
    runner = build_runner(cfg)
    metric = _find_angle_metric(runner.test_evaluator)
    metric.logger = runner.logger
    print_log(f"Using processed results pickle: {metric.results_pkl}", logger=runner.logger)
    add_dump_results_metric(runner, out_path)
    print_log(log_message, logger=runner.logger)
    runner.test()
    return runner


def main():
    start_time = time.time()
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    if args.results_pkl is None:
        args.results_pkl = get_default_results_pkl(cfg.work_dir, args.checkpoint)

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        assert "tta_model" in cfg, "Cannot find ``tta_model`` in config."
        assert "tta_pipeline" in cfg, "Cannot find ``tta_pipeline`` in config."
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    cfg.test_evaluator.checkpoint_path = args.checkpoint
    angle_sectors = resolve_angle_sectors(args)
    cfg = install_angle_metric(cfg, angle_sectors, include_overall=not args.no_overall, args=args)

    if "dataset_test_groups" in cfg:
        for dataset_name, dataset_configs in cfg.dataset_test_groups.items():
            dataset_file, evaluate_frame_prefix = dataset_configs
            cfg.test_dataloader.dataset.ann_file = osp.join(cfg.info_directory_path, dataset_file)
            cfg.test_evaluator.dataset_name = dataset_name
            if cfg.test_evaluator.get("type") == "T4MetricV2":
                cfg.test_evaluator.evaluate_frame_prefix = evaluate_frame_prefix
            cfg.test_evaluator.ann_file = osp.join(cfg.data_root, cfg.info_directory_path, dataset_file)

            results_pkl = get_group_output_path(args.results_pkl, dataset_name)
            log_message = f"Testing dataset: {dataset_name}, file: {dataset_file}, angle_sectors: {list(angle_sectors)}"
            if results_pkl and osp.exists(results_pkl):
                cfg.test_evaluator.results_pkl = results_pkl
                runner = evaluate_from_results_pkl(cfg, results_pkl, log_message)
            else:
                cfg.test_evaluator.results_pkl = results_pkl
                runner = test_and_optionally_save_results(
                    cfg,
                    get_group_output_path(args.out, dataset_name),
                    log_message,
                )
    else:
        log_message = f"Testing angle_sectors: {list(angle_sectors)}"
        if args.results_pkl and osp.exists(args.results_pkl):
            cfg.test_evaluator.results_pkl = args.results_pkl
            runner = evaluate_from_results_pkl(cfg, args.results_pkl, log_message)
        else:
            cfg.test_evaluator.results_pkl = args.results_pkl
            runner = test_and_optionally_save_results(cfg, args.out, log_message)

    elapsed_time = time.time() - start_time
    print_log(f"Elapsed time: {elapsed_time:.4f} seconds", logger=runner.logger)


if __name__ == "__main__":
    main()
