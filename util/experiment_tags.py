import json
import re
from pathlib import Path


TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

DATASET_VARIANTS = {
    "monolines3d_big": "big",
    "monolines3d_big2": "big2",
    "monolines3d_imw2020": "imw2020",
    "monolines3d_minibatch": "minibatch",
    "monolines3d_ml_hypersim": "ml_hypersim",
}

HGNET_SIZES = {
    "HGNetv2_B0": "hgnetv2_n",
    "HGNetv2_B1": "hgnetv2_s",
    "HGNetv2_B2": "hgnetv2_m",
    "HGNetv2_B4": "hgnetv2_l",
}

DECODER_TAGS = {
    "independent": "independent",
    "refline3d": "refined3d",
    "refline3d_conditioned": "refined3d_conditioned",
}


def _get(settings, name, default=None):
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def validate_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]

    result = []
    for tag in tags:
        if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                f"Invalid experiment tag {tag!r}; tags must use lowercase "
                "letters, digits, and underscores."
            )
        result.append(tag)
    return sorted(set(result))


def infer_experiment_tags(settings):
    tags = set()

    dataset_name = _get(settings, "dataset_name")
    if dataset_name:
        tags.add(str(dataset_name).lower())
    dataset_root = _get(settings, "mono3d_dataset_root")
    if dataset_root:
        variant = DATASET_VARIANTS.get(Path(dataset_root).name)
        if variant:
            tags.add(variant)

    is_linea3d = bool(_get(settings, "linea3d", False))
    tags.add("linea3d" if is_linea3d else "linea")

    backbone = _get(settings, "backbone", "")
    if backbone == "mogev2bb":
        tags.add("mogev2bb")
        base_model = str(_get(settings, "mogev2bb_base_model", ""))
        for size in ("vits", "vitb", "vitl"):
            if f"-{size}-" in base_model:
                tags.add(size)
                break
        num_tokens = _get(settings, "mogev2bb_num_tokens")
        if num_tokens is not None:
            tags.add(f"tokens_{int(num_tokens)}")
    elif str(backbone).startswith("HGNetv2"):
        tags.add("hgnetv2")
        size = HGNET_SIZES.get(backbone)
        if size:
            tags.add(size)

    if is_linea3d:
        strategy = _get(settings, "line3d_pred_strategy", "direct") or "direct"
        if strategy in {"direct", "uv_depth"}:
            tags.add(strategy)

        decoder_mode = _get(settings, "line3d_decoder_mode", "independent") or "independent"
        decoder_tag = DECODER_TAGS.get(decoder_mode)
        if decoder_tag:
            tags.add(decoder_tag)

        alignment = _get(settings, "line3d_alignment")
        if alignment in {"z_shift", "xyz_shift"}:
            tags.add(alignment)

        loss_type = _get(settings, "line3d_loss_type", "l1") or "l1"
        if loss_type in {"l1", "smooth_l1"}:
            tags.add(loss_type)

        schedule = _get(settings, "line3d_loss_weight_schedule")
        warmup_steps = _get(settings, "line3d_loss_weight_warmup_steps", 0) or 0
        if schedule not in (None, "constant") or warmup_steps > 0:
            tags.add("loss_weight_warmup")

        if bool(_get(settings, "mono3d_normalize_3d_line_space", False)):
            tags.add("normalized_3d_space")
        invalid_filter = _get(settings, "mono3d_invalid_line_filter")
        if invalid_filter not in (None, False, "none"):
            tags.add("invalid_lines_filtered")

    tags.add("amp" if bool(_get(settings, "amp", False)) else "no_amp")
    if _get(settings, "resume"):
        tags.add("continuation")

    return sorted(tags)


def build_experiment_tags(settings, custom_tags=None):
    if custom_tags is None:
        custom_tags = _get(settings, "experiment_tags", [])
    return sorted(set(infer_experiment_tags(settings)) | set(validate_tags(custom_tags)))


def write_experiment_tags(output_dir, settings, custom_tags=None):
    tags = build_experiment_tags(settings, custom_tags=custom_tags)
    path = Path(output_dir) / "experiment_tags.json"
    path.write_text(json.dumps(tags, indent=2) + "\n")
    return tags


def resolve_experiment_text(config_value, cli_value, field_name, multiline=False):
    value = cli_value if cli_value is not None else config_value
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}.")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if not multiline and ("\n" in value or "\r" in value):
        raise ValueError(f"{field_name} must be a single line.")
    return value


def write_experiment_text_metadata(output_dir, name=None, description=None):
    output_dir = Path(output_dir)
    written = []
    for filename, value in (
        ("experiment_name.txt", name),
        ("experiment_description.txt", description),
    ):
        if value is not None:
            path = output_dir / filename
            path.write_text(value + "\n", encoding="utf-8")
            written.append(path)
    return written
