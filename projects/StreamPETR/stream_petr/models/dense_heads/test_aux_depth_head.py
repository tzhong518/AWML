"""Known-input/known-output tests for AuxDepthHead's Gaussian soft depth loss."""
import torch

from .aux_depth_head import AuxDepthHead


def _make_head(**kwargs):
    defaults = dict(in_channels=4, mid_channels=4, num_depth_bins=10, depth_min=0.0, depth_max=10.0)
    defaults.update(kwargs)
    return AuxDepthHead(**defaults)


def test_soft_target_sums_to_one_and_peaks_at_true_bin():
    head = _make_head(depth_bin_sigma=0.5)
    depth = torch.tensor([3.0, 7.0])  # bin width 1.0 -> exact bin centers 3 and 7
    target = head._depth_to_soft_target(depth)
    assert torch.allclose(target.sum(dim=1), torch.ones(2), atol=1e-6)
    assert target.argmax(dim=1).tolist() == [3, 7]


def test_wider_sigma_spreads_more_mass_to_neighbors():
    depth = torch.tensor([5.0])
    narrow = _make_head(depth_bin_sigma=0.3)._depth_to_soft_target(depth)
    wide = _make_head(depth_bin_sigma=3.0)._depth_to_soft_target(depth)
    # neighbor bin (index 6) should get more relative mass under the wider Gaussian
    assert wide[0, 6] > narrow[0, 6]
    # peak bin should get less relative mass under the wider Gaussian
    assert wide[0, 5] < narrow[0, 5]


def test_out_of_range_depth_saturates_at_edge_bins_instead_of_underflowing():
    # 120 bins over [1.0, 61.2], like the t4 configs. Without clamping, a 100 m label
    # sits ~76 bins past the last one and every Gaussian term underflows to 0 in fp32,
    # normalizing to an all-zero row that silently contributes zero loss.
    head = _make_head(num_depth_bins=120, depth_min=1.0, depth_max=61.2, depth_bin_sigma=1.0)
    depth = torch.tensor([100.0, 0.2])  # far beyond depth_max / below depth_min
    target = head._depth_to_soft_target(depth)
    assert torch.allclose(target.sum(dim=1), torch.ones(2), atol=1e-6)
    assert target.argmax(dim=1).tolist() == [119, 0]


def test_loss_is_lower_for_logits_matching_soft_target():
    head = _make_head(depth_bin_sigma=1.0, loss_weight=1.0)
    feat = torch.zeros(1, 4, 2, 2)
    sparse_depth = torch.full((1, 2, 2), 5.0)
    mask = torch.ones(1, 2, 2)

    with torch.no_grad():
        logits = head.forward(feat)

    # good logits: sharp peak at the true bin (index 5)
    good_logits = torch.full_like(logits, -10.0)
    good_logits[:, 5] = 10.0
    # bad logits: sharp peak at a far bin (index 0)
    bad_logits = torch.full_like(logits, -10.0)
    bad_logits[:, 0] = 10.0

    def loss_from_logits(fixed_logits):
        target = head._depth_to_soft_target(sparse_depth[mask > 0])
        logp = torch.nn.functional.log_softmax(fixed_logits.permute(0, 2, 3, 1)[mask > 0], dim=1)
        return -(target * logp).sum(dim=1).mean()

    assert loss_from_logits(good_logits) < loss_from_logits(bad_logits)


def test_loss_handles_empty_mask_without_nan():
    head = _make_head()
    feat = torch.randn(1, 4, 2, 2, requires_grad=True)
    sparse_depth = torch.zeros(1, 2, 2)
    mask = torch.zeros(1, 2, 2)
    loss = head.loss(feat, sparse_depth, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_upsample_factor_one_keeps_native_resolution():
    head = _make_head(upsample_factor=1)
    feat = torch.randn(1, 4, 3, 5)
    logits = head.forward(feat)
    assert logits.shape[-2:] == (3, 5)
    assert head.upsample is None


def test_upsample_factor_scales_spatial_resolution():
    head = _make_head(upsample_factor=2)
    feat = torch.randn(1, 4, 3, 5)
    logits = head.forward(feat)
    assert logits.shape[-2:] == (6, 10)


def test_loss_runs_with_upsampled_logits_matching_label_resolution():
    head = _make_head(upsample_factor=2)
    feat = torch.randn(1, 4, 3, 5, requires_grad=True)
    # label grid must match the upsampled (6, 10) resolution, not the input (3, 5)
    sparse_depth = torch.full((1, 6, 10), 4.0)
    mask = torch.ones(1, 6, 10)
    loss = head.loss(feat, sparse_depth, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert feat.grad is not None
