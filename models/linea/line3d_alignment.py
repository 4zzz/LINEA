"""Shared endpoint-order resolution for scale/shift-aligned 3D lines."""

import torch


def fit_matched_lines3d_alignment(src_pts, tgt_pts, fit_alignment, regression_loss):
    """Fit one image-level alignment after resolving each line's orientation."""
    src_centers = src_pts.mean(dim=1).unsqueeze(0)
    tgt_centers = tgt_pts.mean(dim=1).unsqueeze(0)
    center_weight = src_pts.new_ones(src_centers.shape[:2])
    scale, shift = fit_alignment(src_centers, tgt_centers, center_weight)

    src_initial = scale[:, None, None] * src_pts.unsqueeze(0) + shift[:, None, None, :]
    src_initial = src_initial.squeeze(0)
    tgt_pts_swapped = tgt_pts[:, [1, 0], :]
    direct_error = regression_loss(src_initial, tgt_pts).sum(dim=(1, 2))
    swapped_error = regression_loss(src_initial, tgt_pts_swapped).sum(dim=(1, 2))
    use_swapped = swapped_error < direct_error
    tgt_pts_ordered = torch.where(use_swapped[:, None, None], tgt_pts_swapped, tgt_pts)

    src_pts_flat = src_pts.reshape(1, -1, 3)
    tgt_pts_flat = tgt_pts_ordered.reshape(1, -1, 3)
    weight = src_pts.new_ones(src_pts_flat.shape[:2])
    scale, shift = fit_alignment(src_pts_flat, tgt_pts_flat, weight)
    return scale, shift, use_swapped
