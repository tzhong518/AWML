# ------------------------------------------------------------------------
# Training-only auxiliary depth supervision head for StreamPETR.
# Never called at inference / ONNX / deploy, so it does not affect runtime speed.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class AuxDepthHead(nn.Module):
    """Lightweight auxiliary depth head.

    ``Conv3x3(in->mid) + ReLU + Conv1x1(mid->num_depth_bins)`` on the neck feature
    (256ch, stride 16). Predicts a per-pixel depth-bin distribution and is supervised
    by the LiDAR-projected sparse depth only on cells where the mask is 1 (depth-bin
    classification is more stable than regression given how sparse the label is; this
    matches the CaDDN / BEVDepth practice).

    Args:
        in_channels (int): channels of the input feature (neck output, 256).
        mid_channels (int): hidden channels of the 3x3 conv.
        num_depth_bins (int): number of uniform depth bins.
        depth_min (float): near plane of the binned depth range (meters, camera-Z).
        depth_max (float): far plane of the binned depth range (meters, camera-Z).
        loss_weight (float): weight applied to the returned depth loss.
    """

    def __init__(
        self,
        in_channels=256,
        mid_channels=64,
        num_depth_bins=80,
        depth_min=1.0,
        depth_max=61.2,
        loss_weight=1.0,
    ):
        super().__init__()
        self.num_depth_bins = int(num_depth_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.loss_weight = float(loss_weight)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.num_depth_bins, kernel_size=1),
        )

    def forward(self, feat):
        """feat: (B*N, C, Hf, Wf) -> logits (B*N, num_depth_bins, Hf, Wf)."""
        return self.conv(feat)

    def _depth_to_bin(self, depth):
        bin_w = (self.depth_max - self.depth_min) / self.num_depth_bins
        idx = torch.floor((depth - self.depth_min) / bin_w).long()
        return idx.clamp(0, self.num_depth_bins - 1)

    def loss(self, feat, sparse_depth, sparse_depth_mask):
        """Compute the masked depth-bin classification loss.

        Args:
            feat (Tensor): (B*N, C, Hf, Wf) neck feature (grad frame only).
            sparse_depth (Tensor): (B*N, Hf, Wf) camera-Z depth label (meters).
            sparse_depth_mask (Tensor): (B*N, Hf, Wf) 1 where a LiDAR point projected.

        Returns:
            Tensor: scalar ``loss_aux_depth``.
        """
        logits = self.forward(feat)  # (B*N, num_bins, Hf, Wf)
        mask = sparse_depth_mask > 0
        if mask.sum() == 0:
            # keep the head in the graph so DDP does not complain about unused params
            return logits.sum() * 0.0

        logits = logits.permute(0, 2, 3, 1)  # (B*N, Hf, Wf, num_bins)
        valid_logits = logits[mask]  # (P, num_bins)
        valid_depth = sparse_depth[mask]  # (P,)
        target = self._depth_to_bin(valid_depth)
        loss = F.cross_entropy(valid_logits, target)
        return loss * self.loss_weight
