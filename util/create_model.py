from pathlib import Path

import torch

class LineaModel(torch.nn.Module):
    def __init__(self, model, postprocessor, raw_outputs: bool):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()
        self.raw_outputs = raw_outputs

    def forward(self, images, orig_target_sizes, targets=None):
        raw_outputs = self.model(images, targets)
        geometry = {}
        if targets is not None and all('image_size_before_padding' in target for target in targets):
            geometry = {
                'image_sizes': torch.stack([
                    target['image_size_before_padding'].flip(0) for target in targets
                ]),
                'padded_sizes': torch.stack([target['size'].flip(0) for target in targets]),
            }
        lines, scores = self.postprocessor(raw_outputs, orig_target_sizes, **geometry)
        if self.raw_outputs:
            return raw_outputs, lines, scores
        return lines, scores


def _create_model(model_args):
    from models.registry import MODULE_BUILD_FUNCS

    class_module = getattr(model_args, "modelname")
    if class_module not in MODULE_BUILD_FUNCS._module_dict:
        raise Exception(f"Unknown model module {class_module!r}.")
    return MODULE_BUILD_FUNCS.get(class_module)(model_args)

def update_model_args(model_args):
    if not hasattr(model_args, "linea3d"):
        model_args.linea3d = False
    if not hasattr(model_args, "line3d_pred_strategy"):
        model_args.line3d_pred_strategy = "direct"
    if str(getattr(model_args, "backbone", "")).startswith("HGNetv2"):
        model_args.pretrained = False
    # for safety output_dir is removed so training info is overwritten by dataset loader
    model_args.output_dir = None

def create_eval_model_from_checkpoint(checkpoint_path: Path | str, raw_outputs: bool = False):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model_args = checkpoint["args"]

    model_meta = {
        'training_output_dir': getattr(model_args, "output_dir", None)
    }

    update_model_args(model_args)
    model_meta['needs_camera_k'] = (
        model_args.linea3d and model_args.line3d_pred_strategy == "uv_depth"
    )

    model, postprocessor = _create_model(model_args)

    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        model_meta['weights_source'] = 'ema'
    else:
        state = checkpoint["model"]
        model_meta['weights_source'] = 'model'

    model.load_state_dict(state)

    return LineaModel(model, postprocessor, raw_outputs).eval(), model_args, model_meta
