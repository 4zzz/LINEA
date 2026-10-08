"""Compact JSON output helpers shared by inference entry points."""

from __future__ import annotations

import json
from pathlib import Path

from util.prediction_record import to_jsonable


def build_simple_prediction(
    lines2d,
    scores,
    index: int,
    threshold: float,
    raw_outputs=None,
    fitted_lines3d=None,
):
    """Build parallel prediction arrays for one batch element."""
    sample_scores = scores[index]
    keep = sample_scores > threshold
    prediction = {
        "scores": sample_scores[keep].detach().cpu().tolist(),
        "lines2d": lines2d[index][keep].detach().cpu().tolist(),
    }
    if raw_outputs is not None and "pred_lines3d" in raw_outputs:
        prediction["lines3d"] = (
            raw_outputs["pred_lines3d"][index][keep].detach().cpu().tolist()
        )
    if fitted_lines3d is not None:
        prediction["lines3d_fitted"] = fitted_lines3d[keep].detach().cpu().tolist()
    return prediction


def save_simple_prediction(path: Path, prediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(prediction), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
