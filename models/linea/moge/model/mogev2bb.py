from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
#import utils3d
from huggingface_hub import hf_hub_download

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, angle_diff_vec3
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from .modules import DINOv2Encoder, MLP, ConvStack

    
class MoGeV2BB(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        num_tokens_range: List[int] = [1200, 3600],
        use_neck: bool = False,
        **deprecated_kwargs
    ):
        super(MoGeV2BB, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.num_tokens_range = num_tokens_range
        self.use_neck = use_neck

        self.encoder = DINOv2Encoder(**encoder) 
        if use_neck:
            self.neck = ConvStack(**neck)
            self.neck_num_levels = len(self.neck.res_blocks)
        else:
            self.neck_num_levels = 1

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        neck_config_override: Optional[Dict[str, Any]] = None,
        **hf_kwargs
    ) -> 'MoGeV2BB':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `neck_config_override`: replaces the checkpoint neck config. Useful when experimenting with a different
          number of neck levels or channel widths.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = copy.deepcopy(checkpoint['model_config'])
        if neck_config_override is not None:
            model_config['neck'] = copy.deepcopy(neck_config_override)
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)

        model_state = model.state_dict()
        checkpoint_state = checkpoint['model']

        compatible_state = {}
        skipped_shape_mismatch = []
        unexpected_keys = []
        for key, value in checkpoint_state.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            compatible_state[key] = value

        load_result = model.load_state_dict(compatible_state, strict=False)

        missing_keys = list(load_result.missing_keys)
        if skipped_shape_mismatch or missing_keys or unexpected_keys:
            mismatch_summary = []
            if skipped_shape_mismatch:
                preview = ', '.join(
                    f"{key}: ckpt{ckpt_shape} -> model{model_shape}"
                    for key, ckpt_shape, model_shape in skipped_shape_mismatch[:8]
                )
                if len(skipped_shape_mismatch) > 8:
                    preview += f", ... (+{len(skipped_shape_mismatch) - 8} more)"
                mismatch_summary.append(f"shape-mismatched keys skipped: {preview}")
            if missing_keys:
                preview = ', '.join(missing_keys[:8])
                if len(missing_keys) > 8:
                    preview += f", ... (+{len(missing_keys) - 8} more)"
                mismatch_summary.append(f"missing keys left at model init: {preview}")
            if unexpected_keys:
                preview = ', '.join(unexpected_keys[:8])
                if len(unexpected_keys) > 8:
                    preview += f", ... (+{len(unexpected_keys) - 8} more)"
                mismatch_summary.append(f"unexpected checkpoint keys ignored: {preview}")
            warnings.warn("Partial MoGeV2BB checkpoint load. " + " | ".join(mismatch_summary))
        
        return model
    
    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        if self.use_neck:
            self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()
    
    def forward(self, image, num_tokens: Union[int, torch.LongTensor]=1200) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        if self.use_neck is False:
            return [features], cls_token

        features = [features] + [None] * (self.neck_num_levels - 1)

        # Concat UVs for aspect ratio input
        for level in range(self.neck_num_levels):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        return features[::-1], cls_token
