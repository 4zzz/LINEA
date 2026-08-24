# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


def _finite_summary(name, tensor):
    finite = torch.isfinite(tensor)
    if finite.any():
        valid = tensor[finite]
        return (
            f"{name}: finite={finite.sum().item()}/{tensor.numel()} "
            f"min={valid.min().item():.6g} max={valid.max().item():.6g}"
        )
    return f"{name}: finite=0/{tensor.numel()}"


def _nonfinite_batch_indices(tensor):
    if tensor.ndim == 0:
        return [0] if not torch.isfinite(tensor) else []
    finite_per_sample = torch.isfinite(tensor).flatten(1).all(dim=1)
    return (~finite_per_sample).nonzero(as_tuple=False).flatten().cpu().tolist()


@torch.no_grad()
def _compute_costs(matcher, outputs, targets):
    """Return the exact per-image cost components used by the matcher."""
    bs, num_queries = outputs["pred_logits"].shape[:2]
    # Matching does not need gradients. Keep its probability and distance
    # calculations in FP32 so autocast cannot saturate sigmoid to exactly 0/1.
    out_prob = outputs["pred_logits"].flatten(0, 1).float().sigmoid()
    out_line = outputs["pred_lines"].flatten(0, 1).float()

    sizes = [len(target["lines"]) for target in targets]
    if sum(sizes) == 0:
        return [
            {
                name: out_line.new_empty((num_queries, 0))
                for name in (
                    "classification_raw",
                    "line2d_raw",
                    "classification_weighted",
                    "line2d_weighted",
                    "total",
                )
            }
            for _ in targets
        ]

    tgt_ids = torch.cat([target["labels"] for target in targets])
    tgt_line = torch.cat([target["lines"] for target in targets]).float()

    alpha = matcher.focal_alpha
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    cost_line = torch.cdist(out_line, tgt_line, p=1)
    weighted_class = matcher.cost_class * cost_class
    weighted_line = matcher.cost_line * cost_line
    total = weighted_line + weighted_class

    if not torch.isfinite(total).all():
        image_ids = [
            target.get("image_id", torch.tensor([-1])).detach().cpu().flatten().tolist()
            for target in targets
        ]
        invalid_batch_indices = sorted(set(
            _nonfinite_batch_indices(outputs["pred_logits"])
            + _nonfinite_batch_indices(outputs["pred_lines"])
        ))
        invalid_image_ids = [image_ids[index] for index in invalid_batch_indices]
        raise ValueError(
            "matcher cost matrix contains non-finite values; "
            + "; ".join([
                _finite_summary("pred_logits", outputs["pred_logits"]),
                _finite_summary("pred_lines", outputs["pred_lines"]),
                _finite_summary("target_lines", tgt_line),
                _finite_summary("cost_class", cost_class),
                _finite_summary("cost_line", cost_line),
                _finite_summary("cost", total),
                f"invalid_batch_indices={invalid_batch_indices}",
                f"invalid_image_ids={invalid_image_ids}",
                f"target_sizes={sizes}",
                f"image_ids={image_ids}",
            ])
        )

    components = {
        "classification_raw": cost_class.view(bs, num_queries, -1),
        "line2d_raw": cost_line.view(bs, num_queries, -1),
        "classification_weighted": weighted_class.view(bs, num_queries, -1),
        "line2d_weighted": weighted_line.view(bs, num_queries, -1),
        "total": total.view(bs, num_queries, -1),
    }
    split_components = {
        name: matrix.split(sizes, dim=-1)
        for name, matrix in components.items()
    }
    return [
        {name: matrices[image_index][image_index] for name, matrices in split_components.items()}
        for image_index in range(bs)
    ]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_line = cost_bbox
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def compute_costs(self, outputs, targets):
        return _compute_costs(self, outputs, targets)

    @torch.no_grad()
    def match_from_costs(self, costs):
        indices = [linear_sum_assignment(cost["total"].cpu()) for cost in costs]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        return self.match_from_costs(self.compute_costs(outputs, targets))


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_line = cost_bbox
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def compute_costs(self, outputs, targets):
        return _compute_costs(self, outputs, targets)

    @torch.no_grad()
    def match_from_costs(self, costs):
        indices = []
        for cost in costs:
            weight_mat = cost["total"]
            if weight_mat.shape[1] == 0:
                empty = torch.empty(0, dtype=torch.int64, device=weight_mat.device)
                indices.append((empty, empty))
                continue
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(weight_mat.shape[1], device=weight_mat.device)
            indices.append((idx_i, idx_j))
        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        return self.match_from_costs(self.compute_costs(outputs, targets))


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_lines, focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))
