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

    ``[optional ConvTranspose2d upsample] + Conv3x3(in->mid) + ReLU +
    Conv1x1(mid->num_depth_bins)`` on the neck feature (256ch, stride 16 by default).
    Predicts a per-pixel depth-bin distribution and is supervised by the
    LiDAR-projected sparse depth only on cells where the mask is 1. The target is a
    Gaussian soft label centered at the true depth bin (std ``depth_bin_sigma`` bins)
    rather than a one-hot bin, so neighboring bins get partial credit and the loss no
    longer penalizes near-miss predictions as harshly as far ones (matches the
    soft-target practice in DID-M3D / SoftGroup-style depth binning).

    ``upsample_factor`` > 1 learnably upsamples the neck feature before predicting,
    so the depth label grid can be finer than the neck's native stride (fewer LiDAR
    points get merged into the same cell by the nearest-depth reduction upstream).
    This does not make the backbone features more spatially precise than their
    receptive field — it only reduces label-merging noise and denses up supervision.

    Args:
        in_channels (int): channels of the input feature (neck output, 256).
        mid_channels (int): hidden channels of the 3x3 conv.
        num_depth_bins (int): number of uniform depth bins.
        depth_min (float): near plane of the binned depth range (meters, camera-Z).
        depth_max (float): far plane of the binned depth range (meters, camera-Z).
        depth_bin_sigma (float): std, in units of bins, of the Gaussian soft target.
        upsample_factor (int): integer upsampling factor applied to the input feature
            before prediction (1 = no upsampling, matches the neck's native stride).
        loss_weight (float): weight applied to the returned depth loss.
    """

    def __init__(
        self,
        in_channels=256,
        mid_channels=64,
        num_depth_bins=80,
        depth_min=1.0,
        depth_max=61.2,
        depth_bin_sigma=1.0,
        upsample_factor=1,
        loss_weight=1.0,
    ):
        super().__init__()
        self.num_depth_bins = int(num_depth_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_bin_sigma = float(depth_bin_sigma)
        self.upsample_factor = int(upsample_factor)
        self.loss_weight = float(loss_weight)

        if self.upsample_factor > 1:
            self.upsample = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=self.upsample_factor,
                stride=self.upsample_factor,
            )
        else:
            self.upsample = None

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.num_depth_bins, kernel_size=1),
        )

    def forward(self, feat):
        """feat: (B*N, C, Hf, Wf) -> logits (B*N, num_depth_bins, Hf', Wf').

        Hf', Wf' == Hf, Wf when ``upsample_factor == 1``, else
        Hf' = Hf * upsample_factor (and likewise for Wf').
        """
        if self.upsample is not None:
            feat = self.upsample(feat)
        return self.conv(feat)

    def _depth_to_soft_target(self, depth):
        """Gaussian soft label over bins, centered at the (fractional) true bin.

        Args:
            depth (Tensor): (P,) camera-Z depth label (meters).

        Returns:
            Tensor: (P, num_depth_bins) target distribution, rows sum to 1.
        """
        bin_w = (self.depth_max - self.depth_min) / self.num_depth_bins
        center = (depth - self.depth_min) / bin_w  # fractional bin index, (P,)
        bin_ids = torch.arange(self.num_depth_bins, device=depth.device, dtype=depth.dtype)
        dist_sq = (bin_ids.unsqueeze(0) - center.unsqueeze(1)) ** 2  # (P, num_bins)
        target = torch.exp(-dist_sq / (2.0 * self.depth_bin_sigma**2))
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return target

    def loss(self, feat, sparse_depth, sparse_depth_mask):
        """Compute the masked depth-bin soft-classification loss.

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
        target = self._depth_to_soft_target(valid_depth)  # (P, num_bins)
        log_prob = F.log_softmax(valid_logits, dim=1)
        loss = -(target * log_prob).sum(dim=1).mean()
        return loss * self.loss_weight
