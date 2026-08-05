import os

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


def project_to_image(points, lidar2cam, cam2img):
    """Transform points from LiDAR to image coordinates."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T

    # Filter points behind the camera
    valid_mask = points_cam[:, 2] > 0

    points_img = np.dot(cam2img, points_cam[:, :3].T).T
    points_img /= points_img[:, 2:3]
    return points_img[:, :2], valid_mask


def _get_img_hw(img_shape):
    if len(img_shape) < 2:
        raise ValueError(f"Invalid image shape: {img_shape}")
    if len(img_shape) == 2:
        return img_shape

    # mmcv image loaders usually return HWC, while some bundle stages use CHW.
    if img_shape[0] in (1, 3) and img_shape[-1] not in (1, 3):
        return img_shape[1], img_shape[2]
    return img_shape[0], img_shape[1]


def compute_bbox_and_centers(lidar2cam, cam2img, bboxes, labels, img_shape):
    """
    Compute the 2D bounding box, 3D center of the projected bounding box, and 3D center in LiDAR coordinates.

    Args:
        data_dict (dict): Contains the image path, lidar2cam, and cam2img transformation matrices.
        bboxes (object): Contains the 3D bounding box corners and labels.
        labels (np.ndarray): Array of labels for each bbox
        img_shape (tuple): Image dimensions (H, W)

    Returns:
        tuple: Contains:
            - bboxes_2d: np.ndarray of shape (N, 4) for [x1, y1, x2, y2]
            - projected_centers: np.ndarray of shape (N, 2) for projected 3D centers
            - centers_3d: np.ndarray of shape (N, 3) for 3D centers in LiDAR coords
            - valid_labels: np.ndarray of shape (N,) containing labels for valid boxes
    """

    H, W = _get_img_hw(img_shape)
    # Initialize lists to store valid results
    valid_bboxes_2d = []
    valid_projected_centers = []
    valid_image_depth = []
    valid_labels_list = []

    # Loop through each bounding box
    for bbox_std, bbox, label in zip(bboxes, bboxes.corners, labels):
        # Project corners to image
        center_3d_lidar = bbox_std[:3].numpy()
        corners_img, valid_mask = project_to_image(
            np.concatenate([bbox, bbox.mean(0).reshape(1, 3)]), lidar2cam, cam2img
        )
        projected_center = corners_img[-1]

        corners_img = corners_img[:-1][valid_mask[:-1]]

        if len(corners_img) == 0:  # Skip if no corners are visible
            continue

        # Compute 2D bbox
        x_min, y_min = np.min(corners_img, axis=0)
        x_max, y_max = np.max(corners_img, axis=0)

        # Clip to image boundaries
        x_min = np.clip(x_min, 0, W)
        x_max = np.clip(x_max, 0, W)
        y_min = np.clip(y_min, 0, H)
        y_max = np.clip(y_max, 0, H)

        x_center = np.clip(projected_center[0], 0, W)
        y_center = np.clip(projected_center[1], 0, H)
        if x_min == x_max or y_min == y_max:
            continue

        valid_bboxes_2d.append([x_min, y_min, x_max, y_max])
        valid_projected_centers.append([x_center, y_center])
        valid_image_depth.append(np.sqrt((center_3d_lidar**2).sum()))
        valid_labels_list.append(label)

    if valid_bboxes_2d:
        bboxes_2d = np.array(valid_bboxes_2d)
        projected_centers = np.array(valid_projected_centers)
        object_depth = np.array(valid_image_depth)
        valid_labels = np.array(valid_labels_list)
    else:
        # Return empty arrays with correct shapes if no valid boxes
        bboxes_2d = np.zeros((0, 4))
        projected_centers = np.zeros((0, 2))
        object_depth = np.zeros((0,))
        valid_labels = np.zeros(0, dtype=int)

    return bboxes_2d, projected_centers, object_depth, valid_labels


def check_bbox_visibility_in_image(
    lidar2cam,
    cam2img,
    bboxes,
    img_shape,
    visibility=0.1,
    min_visible_area=0,
    ego_mask_polygon=None,
    return_masked=False,
):
    """
    Projects 3D bounding boxes into the image plane and determines visibility.

    Args:
        lidar2cam (np.ndarray): 4x4 transformation matrix from LiDAR to camera coordinates.
        cam2img (np.ndarray): 3x3 camera intrinsic matrix.
        bboxes (list): List of 3D bounding boxes. Each must have `.corners` attribute and be indexable.
        img_shape (tuple): Shape of the image in HWC or CHW format.
        visibility (float, optional): Minimum fraction (0-1) of projected 2D bbox area that must lie
            within the image to consider it visible. Defaults to 0.1.
        min_visible_area (float, optional): Minimum clipped image area in pixels. A box is kept when either
            visible ratio or visible area passes the configured threshold. Defaults to 0.
        ego_mask_polygon (np.ndarray, optional): Pixel-space polygon. If the clipped projected 2D bbox is
            fully inside this polygon, the box is treated as invisible in this camera.
        return_masked (bool): Also return whether each box was suppressed by the ego mask.

    Returns:
        list: A list of booleans indicating if each bounding box is sufficiently visible.
    """
    H, W = _get_img_hw(img_shape)
    is_visible = []
    is_ego_masked = []

    for bbox in bboxes.corners:
        all_points = np.concatenate([bbox, bbox.mean(0).reshape(1, 3)], axis=0)
        corners_img, valid_mask = project_to_image(all_points, lidar2cam, cam2img)
        corners_img = corners_img[:-1][valid_mask[:-1]]

        if len(corners_img) == 0:
            is_visible.append(False)
            is_ego_masked.append(False)
            continue

        # Compute full 2D bbox from all projected corners
        x_min, y_min = np.min(corners_img, axis=0)
        x_max, y_max = np.max(corners_img, axis=0)
        full_area = max(x_max - x_min, 0) * max(y_max - y_min, 0)

        if full_area == 0:
            is_visible.append(False)
            is_ego_masked.append(False)
            continue

        # Compute clipped bbox (intersection with image frame)
        x_min_clip = np.clip(x_min, 0, W)
        x_max_clip = np.clip(x_max, 0, W)
        y_min_clip = np.clip(y_min, 0, H)
        y_max_clip = np.clip(y_max, 0, H)
        visible_area = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)

        visible_ratio = visible_area / full_area
        visible = visible_ratio >= visibility or visible_area >= min_visible_area
        ego_masked = False
        if visible and ego_mask_polygon is not None:
            clipped_bbox = np.array(
                [
                    [x_min_clip, y_min_clip],
                    [x_max_clip, y_min_clip],
                    [x_max_clip, y_max_clip],
                    [x_min_clip, y_max_clip],
                ],
                dtype=np.float32,
            )
            ego_masked = _points_inside_polygon(clipped_bbox, ego_mask_polygon).all()
            if ego_masked:
                visible = False

        is_visible.append(visible)
        is_ego_masked.append(bool(ego_masked))

    if return_masked:
        return is_visible, is_ego_masked
    return is_visible


def _normalize_mask_polygon(polygon):
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3:
        raise ValueError("ego mask polygon requires at least 3 normalized x/y points")
    if np.any((polygon < 0.0) | (polygon > 1.0)):
        raise ValueError("ego mask polygon values must be normalized to [0, 1]")
    return polygon


def _mask_polygon_to_pixels(polygon, width, height):
    pixels = polygon.copy()
    pixels[:, 0] *= width - 1
    pixels[:, 1] *= height - 1
    return np.rint(pixels).astype(np.int32)


def _points_inside_polygon(points, polygon):
    try:
        import cv2

        polygon = np.asarray(polygon, dtype=np.float32)
        return np.asarray([cv2.pointPolygonTest(polygon, tuple(point), False) >= 0 for point in points], dtype=bool)
    except ImportError:
        x = points[:, 0]
        y = points[:, 1]
        poly_x = polygon[:, 0]
        poly_y = polygon[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        j = len(polygon) - 1
        for i in range(len(polygon)):
            crosses = ((poly_y[i] > y) != (poly_y[j] > y)) & (
                x <= (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-12) + poly_x[i]
            )
            inside ^= crosses
            j = i
        return inside


def _normalize_ego_masks(ego_masks):
    """Validate a {group: {camera: flat_xy_list}} ego-mask config into numpy polygons."""
    normalized = {}
    for group, camera_masks in (ego_masks or {}).items():
        normalized[str(group).lower()] = {
            str(camera): _normalize_mask_polygon(polygon) for camera, polygon in camera_masks.items()
        }
    return normalized


def _infer_ego_mask_group(results, ego_masks):
    """Match an ego-mask group key (e.g. "j6gen2") against sample/image path strings."""
    if not ego_masks:
        return None

    parts = []
    for key in ("frame_idx", "sample_idx", "scene_token"):
        value = results.get(key)
        if value is not None:
            parts.append(str(value))
    parts.extend(str(path) for path in results.get("img_filename", []))
    for cam_info in results.get("images", {}).values():
        if isinstance(cam_info, dict):
            for key in ("img_path", "sample_data_path", "filename"):
                value = cam_info.get(key)
                if value is not None:
                    parts.append(str(value))

    haystack = " ".join(parts).lower()
    for group in ego_masks:
        if group in haystack:
            return group
    return None


def _polygon_cell_mask(polygon_cells, Hf, Wf):
    """Rasterize a polygon given in feature-cell coordinates into a (Hf, Wf) bool mask.

    Boundary cells (partially covered by the polygon) are included, so the mask is
    conservative: supervision is dropped on every cell the polygon touches.
    """
    polygon_cells = np.asarray(polygon_cells, dtype=np.float32)
    try:
        import cv2

        mask = np.zeros((Hf, Wf), dtype=np.uint8)
        pts = np.rint(polygon_cells).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        cv2.polylines(mask, [pts], isClosed=True, color=1, thickness=1)
        return mask.astype(bool)
    except ImportError:
        xs = (np.arange(Wf, dtype=np.float32) + 0.5)[None, :].repeat(Hf, axis=0)
        ys = (np.arange(Hf, dtype=np.float32) + 0.5)[:, None].repeat(Wf, axis=1)
        centers = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
        return _points_inside_polygon(centers, polygon_cells).reshape(Hf, Wf)


def _paint_polygon_mask(img, polygon, color):
    try:
        import cv2
    except ImportError:
        return img

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        hwc_img = np.transpose(img, (1, 2, 0))
        transposed = True
    else:
        hwc_img = img
        transposed = False

    height, width = hwc_img.shape[:2]
    pixel_polygon = _mask_polygon_to_pixels(polygon, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixel_polygon], 255)
    fill = np.asarray(color, dtype=hwc_img.dtype)
    hwc_img[mask == 255] = fill

    if transposed:
        img[...] = np.transpose(hwc_img, (2, 0, 1))
    return img


def _to_hwc_uint8(img):
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _project_box_corners(lidar2cam, cam2img, bbox):
    corners_img, valid_mask = project_to_image(bbox, lidar2cam, cam2img)
    return corners_img, valid_mask


def _visible_ratio_and_area(lidar2cam, cam2img, bbox, img_shape):
    H, W = _get_img_hw(img_shape)
    all_points = np.concatenate([bbox, bbox.mean(0).reshape(1, 3)], axis=0)
    corners_img, valid_mask = project_to_image(all_points, lidar2cam, cam2img)
    corners_img = corners_img[:-1][valid_mask[:-1]]
    if len(corners_img) == 0:
        return 0.0, 0.0

    x_min, y_min = np.min(corners_img, axis=0)
    x_max, y_max = np.max(corners_img, axis=0)
    full_area = max(x_max - x_min, 0) * max(y_max - y_min, 0)
    if full_area == 0:
        return 0.0, 0.0

    x_min_clip = np.clip(x_min, 0, W)
    x_max_clip = np.clip(x_max, 0, W)
    y_min_clip = np.clip(y_min, 0, H)
    y_max_clip = np.clip(y_max, 0, H)
    visible_area = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    return visible_area / full_area, visible_area


def _draw_projected_box(img, corners_img, valid_mask, color, text):
    try:
        import cv2
    except ImportError:
        return img

    H, W = img.shape[:2]
    valid_corners = corners_img[valid_mask]
    if len(valid_corners) == 0:
        return img

    valid_corners[:, 0] = np.clip(valid_corners[:, 0], 0, W - 1)
    valid_corners[:, 1] = np.clip(valid_corners[:, 1], 0, H - 1)
    x_min, y_min = valid_corners.min(axis=0).astype(int)
    x_max, y_max = valid_corners.max(axis=0).astype(int)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    corners = corners_img.copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, W - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, H - 1)
    corners = corners.astype(int)
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edge_indices:
        if valid_mask[start] and valid_mask[end]:
            cv2.line(img, tuple(corners[start]), tuple(corners[end]), color, 1)

    cv2.putText(
        img,
        text,
        (x_min, max(y_min - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return img


def _make_image_grid(images, pad=8):
    if not images:
        return None
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    channels = images[0].shape[2]
    rows = []
    for row_start in range(0, len(images), 3):
        row_imgs = images[row_start : row_start + 3]
        row = np.zeros((max_h, len(row_imgs) * max_w + (len(row_imgs) - 1) * pad, channels), dtype=np.uint8)
        x = 0
        for img in row_imgs:
            row[: img.shape[0], x : x + img.shape[1]] = img
            x += max_w + pad
        rows.append(row)
    grid = np.zeros((len(rows) * max_h + (len(rows) - 1) * pad, rows[0].shape[1], channels), dtype=np.uint8)
    y = 0
    for row in rows:
        grid[y : y + row.shape[0], : row.shape[1]] = row
        y += max_h + pad
    return grid


@TRANSFORMS.register_module()
class LoadSparseDepthFromLiDAR(BaseTransform):
    """Project the LiDAR point cloud onto every camera to build a sparse depth map.

    Training-only auxiliary supervision. Must run AFTER ``ResizeCropFlipRotImage``
    (so ``intrinsics``/``extrinsics`` already reflect the 2D image augmentation) and
    BEFORE ``GlobalRotScaleTransImage`` (which virtually rotates/scales the LiDAR frame
    via ``lidar2img @ R_inv`` while leaving the real image pixels and the point cloud
    untouched). At this point camera-Z of each projected point is the true physical
    depth of the corresponding image pixel, so the label aligns with what the network
    actually sees. Also must run BEFORE ``PadMultiViewImage`` so the depth-map size is
    derived from the unpadded image (480x640 -> 30x40 at stride 16).

    Produces (per camera, at feature resolution H/stride x W/stride):
        results["sparse_depth"]:      (N_cam, Hf, Wf) float32, camera-Z in meters (0 where empty)
        results["sparse_depth_mask"]: (N_cam, Hf, Wf) float32, 1 where a point projected, else 0
    When several points fall in the same feature cell, the nearest (min-Z) is kept.

    ``ego_masks`` (same format as ``Filter3DBoxesinBlindSpot``: {group: {camera:
    flat normalized x/y polygon}}) zeroes the depth label inside the ego-vehicle
    image regions. Those pixels show the ego body (or are painted black by the
    blind-spot transform), so LiDAR returns there would train the head to predict
    depth from uninformative pixels. Polygons are defined on the ORIGINAL image;
    they are mapped through the recorded ``ida_mats`` (resize/crop/flip) when
    ``ResizeCropFlipRotImage`` ran earlier in the pipeline.
    """

    def __init__(self, stride=16, load_dim=5, min_depth=0.1, max_depth=None, ego_masks=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = int(stride)
        self.load_dim = int(load_dim)
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.ego_masks = _normalize_ego_masks(ego_masks)
        self._warned_unknown_ego_mask_group = False

    def _ego_cell_mask(self, results, group, cam_name, cam_idx, W, H, Hf, Wf):
        """Bool (Hf, Wf) mask of feature cells covered by the ego polygon, or None."""
        polygon = self.ego_masks.get(group, {}).get(cam_name) if group else None
        if polygon is None:
            return None
        pts = polygon.copy()
        if "ida_mats" in results:
            # Polygon is normalized on the original image; replay the image aug.
            H0, W0 = results["img_shape_before_ida"]
            pts[:, 0] *= W0 - 1
            pts[:, 1] *= H0 - 1
            ida = np.asarray(results["ida_mats"][cam_idx], dtype=np.float32)
            pts = pts @ ida[:2, :2].T + ida[:2, 2]
        else:
            # No image aug ran; normalized coords apply to the current image directly.
            pts[:, 0] *= W - 1
            pts[:, 1] *= H - 1
        return _polygon_cell_mask(pts / self.stride, Hf, Wf)

    def _load_points(self, pts_filename):
        if not os.path.exists(pts_filename):
            raise FileNotFoundError(f"[LoadSparseDepthFromLiDAR] point cloud not found: {pts_filename}")
        points = np.fromfile(pts_filename, dtype=np.float32)
        points = points.reshape(-1, self.load_dim)[:, :3]
        return points

    def transform(self, results):
        points = self._load_points(results["pts_filename"])
        points_hom = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))

        ego_mask_group = _infer_ego_mask_group(results, self.ego_masks)
        if self.ego_masks and ego_mask_group is None and not self._warned_unknown_ego_mask_group:
            print(
                "[LoadSparseDepthFromLiDAR] ego_masks configured, but no group key matched "
                "sample/image paths; skip ego masking for unmatched samples."
            )
            self._warned_unknown_ego_mask_group = True
        cam_names = list(results.get("images", {}))

        sparse_depth = []
        sparse_depth_mask = []
        for i in range(len(results["intrinsics"])):
            lidar2cam = np.asarray(results["extrinsics"][i], dtype=np.float64)
            cam2img = np.asarray(results["intrinsics"][i], dtype=np.float64)
            H, W = _get_img_hw(results["img"][i].shape)
            Hf, Wf = H // self.stride, W // self.stride

            points_cam = (lidar2cam @ points_hom.T).T  # (M, 4)
            z = points_cam[:, 2]
            pixels = (cam2img[:3, :3] @ points_cam[:, :3].T).T  # (M, 3)
            u = pixels[:, 0] / pixels[:, 2]
            v = pixels[:, 1] / pixels[:, 2]

            valid = z > self.min_depth
            if self.max_depth is not None:
                valid &= z < self.max_depth
            valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

            depth_map = np.full(Hf * Wf, np.inf, dtype=np.float32)
            if valid.any():
                cell_u = np.clip((u[valid] / self.stride).astype(np.int64), 0, Wf - 1)
                cell_v = np.clip((v[valid] / self.stride).astype(np.int64), 0, Hf - 1)
                flat_idx = cell_v * Wf + cell_u
                # keep nearest depth per cell
                np.minimum.at(depth_map, flat_idx, z[valid].astype(np.float32))

            mask = np.isfinite(depth_map)
            depth_map[~mask] = 0.0
            depth_map = depth_map.reshape(Hf, Wf)
            mask = mask.reshape(Hf, Wf)

            if ego_mask_group is not None and i < len(cam_names):
                ego_cells = self._ego_cell_mask(results, ego_mask_group, cam_names[i], i, W, H, Hf, Wf)
                if ego_cells is not None:
                    depth_map[ego_cells] = 0.0
                    mask[ego_cells] = False

            sparse_depth.append(depth_map)
            sparse_depth_mask.append(mask.astype(np.float32))

        results["sparse_depth"] = np.stack(sparse_depth).astype(np.float32)
        results["sparse_depth_mask"] = np.stack(sparse_depth_mask).astype(np.float32)
        return results


@TRANSFORMS.register_module()
class StreamPETRLoadAnnotations2D(BaseTransform):

    def transform(self, results):

        all_bboxes_2d, all_centers_2d, all_depths, all_labels = [], [], [], []

        for i, k in enumerate(results["images"]):
            bboxes_2d, projected_centers, depths, valid_labels = compute_bbox_and_centers(
                results["extrinsics"][i],
                results["intrinsics"][i],
                results["gt_bboxes_3d"],
                results["gt_labels_3d"],
                results["img"][i].shape,
            )
            all_bboxes_2d.append(bboxes_2d)
            all_centers_2d.append(projected_centers)
            all_depths.append(depths)
            all_labels.append(valid_labels)
        results["depths"] = all_depths
        results["centers_2d"] = all_centers_2d
        results["gt_bboxes"] = all_bboxes_2d
        results["gt_bboxes_labels"] = all_labels
        return results


@TRANSFORMS.register_module()
class Filter3DBoxesinBlindSpot(BaseTransform):

    def __init__(
        self,
        visibility=0.1,
        debug_vis_dir=None,
        min_visible_area=0,
        min_lidar_points=0,
        debug_vis_interval=1,
        debug_vis_max_samples=100,
        debug_vis_kept=True,
        debug_log_interval=1,
        debug_log_max_samples=200,
        class_names=None,
        ego_masks=None,
        mask_images=False,
        mask_color=(0, 0, 0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.visibility = visibility
        self.min_visible_area = min_visible_area
        self.min_lidar_points = int(min_lidar_points)
        self.debug_vis_dir = debug_vis_dir
        self.debug_vis_interval = max(int(debug_vis_interval), 1)
        self.debug_vis_max_samples = debug_vis_max_samples
        self.debug_vis_kept = debug_vis_kept
        self.debug_log_interval = max(int(debug_log_interval), 1)
        self.debug_log_max_samples = debug_log_max_samples
        self.class_names = class_names
        self.ego_masks = self._normalize_ego_masks(ego_masks or {})
        self.mask_images = mask_images
        self.mask_color = tuple(mask_color)
        self._num_seen = 0
        self._num_visualized = 0
        self._num_removed_seen = 0
        self._warned_missing_visibility_level = False
        self._num_logged = 0
        self._warned_missing_num_lidar_pts = False
        self._warned_unknown_ego_mask_group = False

    def _label_name(self, label):
        label = int(label)
        if self.class_names is None or label >= len(self.class_names):
            return str(label)
        return self.class_names[label]

    def _normalize_ego_masks(self, ego_masks):
        return _normalize_ego_masks(ego_masks)

    def _infer_ego_mask_group(self, results):
        if not self.ego_masks:
            return None
        group = _infer_ego_mask_group(results, self.ego_masks)
        if group is not None:
            return group
        if not self._warned_unknown_ego_mask_group:
            print(
                "[BlindSpotFilter] ego_masks configured, but no group key matched sample/image paths; "
                "skip ego masking for unmatched samples."
            )
            self._warned_unknown_ego_mask_group = True
        return None

    def _apply_ego_image_masks(self, results, ego_mask_group):
        if not self.mask_images or ego_mask_group is None:
            return
        group_masks = self.ego_masks.get(ego_mask_group, {})
        for cam_idx, cam_name in enumerate(results["images"]):
            polygon = group_masks.get(cam_name)
            if polygon is None:
                continue
            results["img"][cam_idx] = _paint_polygon_mask(results["img"][cam_idx], polygon, self.mask_color)

    def _get_ego_mask_polygon(self, ego_mask_group, cam_name, img_shape):
        if ego_mask_group is None:
            return None
        polygon = self.ego_masks.get(ego_mask_group, {}).get(cam_name)
        if polygon is None:
            return None
        height, width = _get_img_hw(img_shape)
        return _mask_polygon_to_pixels(polygon, width, height).astype(np.float32)

    def _should_visualize(self, removed_indices):
        if self.debug_vis_dir is None or len(removed_indices) == 0:
            return False
        self._num_removed_seen += 1
        if self.debug_vis_max_samples is not None and self._num_visualized >= self.debug_vis_max_samples:
            return False
        return self._num_removed_seen % self.debug_vis_interval == 0

    def _format_label_hist(self, labels):
        hist = {}
        for label in labels:
            name = self._label_name(label)
            hist[name] = hist.get(name, 0) + 1
        return hist

    def _get_num_lidar_pts(self, results, expected_count):
        num_lidar_pts = results.get("num_lidar_pts")
        if num_lidar_pts is None:
            num_lidar_pts = results.get("ann_info", {}).get("num_lidar_pts")
        if num_lidar_pts is None:
            if self.min_lidar_points > 0 and not self._warned_missing_num_lidar_pts:
                print("[BlindSpotFilter] min_lidar_points is enabled, but num_lidar_pts was not found; skip point-count filtering.")
                self._warned_missing_num_lidar_pts = True
            return None

        num_lidar_pts = np.asarray(num_lidar_pts)
        if len(num_lidar_pts) != expected_count:
            if self.min_lidar_points > 0 and not self._warned_missing_num_lidar_pts:
                print(
                    "[BlindSpotFilter] num_lidar_pts length mismatch after earlier GT filtering: "
                    f"expected {expected_count}, got {len(num_lidar_pts)}; skip point-count filtering."
                )
                self._warned_missing_num_lidar_pts = True
            return None
        return num_lidar_pts

    def _log_filter_result(self, results, final_mask, visibility_mask, points_mask, ego_mask=None):
        if self.debug_log_max_samples is not None and self._num_logged >= self.debug_log_max_samples:
            return
        before = len(final_mask)
        kept = int(final_mask.sum())
        removed = before - kept
        lowpts = int((~points_mask).sum())
        blindspot = int((~visibility_mask).sum())
        ego_masked = 0 if ego_mask is None else int(ego_mask.sum())
        should_log = kept == 0 or removed > 0 or self._num_seen % self.debug_log_interval == 0
        if not should_log:
            return

        labels = results["gt_labels_3d"]
        kept_labels = labels[final_mask]
        removed_labels = labels[~final_mask]
        sample_id = (
            results.get("frame_idx")
            or results.get("sample_idx")
            or results.get("img_metas", {}).get("sample_token", "unknown")
        )
        status = results.get("traffic_cone_barrier_status", results.get("img_metas", {}).get("traffic_cone_barrier_status", "unknown"))
        print(
            "[BlindSpotFilter] "
            f"sample={sample_id} status={status} before={before} kept={kept} removed={removed} "
            f"blindspot={blindspot} ego_masked={ego_masked} lowpts={lowpts} "
            f"kept_hist={self._format_label_hist(kept_labels)} "
            f"removed_hist={self._format_label_hist(removed_labels)}"
        )
        self._num_logged += 1

    def _visualize_filter_result(
        self, results, per_cam_visibility, visibility_mask, points_mask, final_mask, num_lidar_pts, per_cam_ego_masked=None
    ):
        try:
            import cv2
        except ImportError:
            return

        removed_indices = np.where(~final_mask)[0]
        if not self._should_visualize(removed_indices):
            return

        os.makedirs(self.debug_vis_dir, exist_ok=True)
        labels = results["gt_labels_3d"]
        bboxes = results["gt_bboxes_3d"]
        images = []

        for cam_idx, cam_name in enumerate(results["images"]):
            img = _to_hwc_uint8(results["img"][cam_idx]).copy()
            for box_idx, bbox in enumerate(bboxes.corners):
                is_removed = not final_mask[box_idx]
                if not is_removed and not self.debug_vis_kept:
                    continue
                lowpts_removed = not points_mask[box_idx]
                blindspot_removed = not visibility_mask[box_idx]
                ego_removed = per_cam_ego_masked is not None and bool(per_cam_ego_masked[:, box_idx].any())
                if lowpts_removed and blindspot_removed:
                    color = (255, 0, 255)
                    tag = "DROP+LOWPTS"
                elif lowpts_removed:
                    color = (0, 0, 255)
                    tag = "LOWPTS"
                elif ego_removed and blindspot_removed:
                    color = (255, 128, 0)
                    tag = "EGO_MASK"
                elif blindspot_removed:
                    color = (0, 255, 255)
                    tag = "DROP"
                else:
                    color = (0, 180, 0)
                    tag = "KEEP"
                corners_img, valid_mask = _project_box_corners(
                    results["extrinsics"][cam_idx],
                    results["intrinsics"][cam_idx],
                    bbox,
                )
                visible_flags = "".join(str(int(v)) for v in per_cam_visibility[:, box_idx])
                ratio, area = _visible_ratio_and_area(
                    results["extrinsics"][cam_idx],
                    results["intrinsics"][cam_idx],
                    bbox,
                    results["img"][cam_idx].shape,
                )
                pts_text = "?" if num_lidar_pts is None else str(int(num_lidar_pts[box_idx]))
                text = f"{tag} {box_idx}:{self._label_name(labels[box_idx])} pts={pts_text} v={visible_flags} r={ratio:.2f} a={area:.0f}"
                _draw_projected_box(img, corners_img, valid_mask, color, text)
            cv2.putText(
                img,
                str(cam_name),
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            images.append(img)

        grid = _make_image_grid(images)
        if grid is None:
            return

        sample_id = (
            results.get("frame_idx")
            or results.get("sample_idx")
            or results.get("img_metas", {}).get("sample_token", "unknown")
        )
        safe_sample_id = str(sample_id).replace("/", "_")
        filename = f"{self._num_seen:06d}_{safe_sample_id}_removed{len(removed_indices)}.jpg"
        output_path = os.path.join(self.debug_vis_dir, filename)
        if cv2.imwrite(output_path, grid):
            print(f"[BlindSpotFilter] saved debug visualization: {output_path}")
            self._num_visualized += 1
        else:
            print(f"[BlindSpotFilter] failed to save debug visualization: {output_path}")

    def transform(self, results):
        self._num_seen += 1
        ego_mask_group = self._infer_ego_mask_group(results)
        self._apply_ego_image_masks(results, ego_mask_group)

        visibility_mask = []
        ego_masked = []
        for i, cam_name in enumerate(results["images"]):
            ego_mask_polygon = self._get_ego_mask_polygon(ego_mask_group, cam_name, results["img"][i].shape)
            is_visible, is_ego_masked = check_bbox_visibility_in_image(
                results["extrinsics"][i],
                results["intrinsics"][i],
                results["gt_bboxes_3d"],
                results["img"][i].shape,
                visibility=self.visibility,
                min_visible_area=self.min_visible_area,
                ego_mask_polygon=ego_mask_polygon,
                return_masked=True,
            )
            visibility_mask.append(is_visible)
            ego_masked.append(is_ego_masked)
        per_cam_visibility = np.stack(visibility_mask)
        per_cam_ego_masked = np.stack(ego_masked)
        visibility_mask = per_cam_visibility.mean(0) > 0  # visible in at least one unmasked view
        ego_mask = per_cam_ego_masked.any(0)
        num_lidar_pts = self._get_num_lidar_pts(results, len(visibility_mask))
        if self.min_lidar_points > 0 and num_lidar_pts is not None:
            points_mask = num_lidar_pts >= self.min_lidar_points
        else:
            points_mask = np.ones_like(visibility_mask, dtype=bool)
        final_mask = visibility_mask & points_mask
        self._log_filter_result(results, final_mask, visibility_mask, points_mask, ego_mask=ego_mask)
        self._visualize_filter_result(
            results,
            per_cam_visibility,
            visibility_mask,
            points_mask,
            final_mask,
            num_lidar_pts,
            per_cam_ego_masked=per_cam_ego_masked,
        )
        results["gt_bboxes_3d"] = results["gt_bboxes_3d"][final_mask]
        results["gt_labels_3d"] = results["gt_labels_3d"][final_mask]
        return results


@TRANSFORMS.register_module()
class Filter3DBoxesByVisibilityLevel(BaseTransform):
    """Filter 3D boxes by T4 annotation visibility level."""

    def __init__(
        self,
        ignored_levels=("none",),
        strict=False,
        debug_vis_dir=None,
        debug_vis_interval=1,
        debug_vis_max_samples=100,
        debug_vis_kept=True,
        class_names=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignored_levels = {str(level) for level in ignored_levels}
        self.strict = strict
        self.debug_vis_dir = debug_vis_dir
        self.debug_vis_interval = max(int(debug_vis_interval), 1)
        self.debug_vis_max_samples = debug_vis_max_samples
        self.debug_vis_kept = debug_vis_kept
        self.class_names = class_names
        self._num_seen = 0
        self._num_visualized = 0
        self._num_removed_seen = 0
        self._warned_missing_visibility_level = False

    def _label_name(self, label):
        label = int(label)
        if self.class_names is None or label >= len(self.class_names):
            return str(label)
        return self.class_names[label]

    def _should_visualize(self, keep_mask):
        if self.debug_vis_dir is None or keep_mask.all():
            return False
        self._num_removed_seen += 1
        if self.debug_vis_max_samples is not None and self._num_visualized >= self.debug_vis_max_samples:
            return False
        return self._num_removed_seen % self.debug_vis_interval == 0

    def _visualize_filter_result(self, results, visibility_level, keep_mask):
        try:
            import cv2
        except ImportError:
            return

        if not self._should_visualize(keep_mask):
            return

        os.makedirs(self.debug_vis_dir, exist_ok=True)
        labels = results["gt_labels_3d"]
        bboxes = results["gt_bboxes_3d"]
        images = []

        for cam_idx, cam_name in enumerate(results["images"]):
            img = _to_hwc_uint8(results["img"][cam_idx]).copy()
            for box_idx, bbox in enumerate(bboxes.corners):
                is_kept = keep_mask[box_idx]
                if is_kept and not self.debug_vis_kept:
                    continue
                color = (0, 180, 0) if is_kept else (0, 0, 255)
                tag = "KEEP" if is_kept else "DROP_VIS_NONE"
                corners_img, valid_mask = _project_box_corners(
                    results["extrinsics"][cam_idx],
                    results["intrinsics"][cam_idx],
                    bbox,
                )
                text = (
                    f"{tag} {box_idx}:{self._label_name(labels[box_idx])} "
                    f"vis={visibility_level[box_idx]}"
                )
                _draw_projected_box(img, corners_img, valid_mask, color, text)
            cv2.putText(
                img,
                str(cam_name),
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            images.append(img)

        grid = _make_image_grid(images)
        if grid is None:
            return

        sample_id = (
            results.get("frame_idx")
            or results.get("sample_idx")
            or results.get("img_metas", {}).get("sample_token", "unknown")
        )
        safe_sample_id = str(sample_id).replace("/", "_")
        removed = int((~keep_mask).sum())
        filename = f"{self._num_seen:06d}_{safe_sample_id}_visibility_removed{removed}.jpg"
        output_path = os.path.join(self.debug_vis_dir, filename)
        if cv2.imwrite(output_path, grid):
            print(f"[VisibilityLevelFilter] saved debug visualization: {output_path}")
            self._num_visualized += 1

    def transform(self, results):
        self._num_seen += 1
        visibility_level = results.get("ann_info", {}).get("visibility_level")
        if visibility_level is None:
            if self.strict:
                raise KeyError("Filter3DBoxesByVisibilityLevel requires `ann_info[\"visibility_level\"]`.")
            if not self._warned_missing_visibility_level:
                print(
                    "[VisibilityLevelFilter] visibility_level is missing; skip filtering. "
                    "Regenerate info pkls with visibility_level to enable this filter."
                )
                self._warned_missing_visibility_level = True
            return results

        visibility_level = np.asarray(visibility_level).astype(str)
        if len(visibility_level) != len(results["gt_labels_3d"]):
            raise ValueError(
                "visibility_level length mismatch: "
                f"expected {len(results['gt_labels_3d'])}, got {len(visibility_level)}"
            )

        keep_mask = np.asarray([level not in self.ignored_levels for level in visibility_level], dtype=bool)
        self._visualize_filter_result(results, visibility_level, keep_mask)
        results["gt_bboxes_3d"] = results["gt_bboxes_3d"][keep_mask]
        results["gt_labels_3d"] = results["gt_labels_3d"][keep_mask]
        for key, value in results.get("ann_info", {}).items():
            if key in ("instances", "gt_bboxes_3d", "gt_labels_3d"):
                continue
            if isinstance(value, np.ndarray) and len(value) == len(keep_mask):
                results["ann_info"][key] = value[keep_mask]
            elif isinstance(value, list) and len(value) == len(keep_mask):
                results["ann_info"][key] = [item for item, keep in zip(value, keep_mask) if keep]
        return results


@TRANSFORMS.register_module()
class Filter3DBoxesByVisibilityToken(Filter3DBoxesByVisibilityLevel):
    """Backward-compatible alias. Prefer Filter3DBoxesByVisibilityLevel."""

    def __init__(self, ignored_tokens=None, ignored_levels=("none",), *args, **kwargs):
        if ignored_tokens is not None:
            print(
                "[VisibilityLevelFilter] ignored_tokens is deprecated and ignored; "
                "filtering uses visibility_level instead."
            )
        super().__init__(ignored_levels=ignored_levels, *args, **kwargs)
