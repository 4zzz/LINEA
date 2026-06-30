from typing import List
import torch
from torch import nn

from .moge.model.mogev2bb import MoGeV2BB

def build_mogev2_backbone(args):

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()

            moge_args = {
                'pretrained_model_name_or_path': args.mogev2bb_base_model,
                'model_kwargs': {
                    'use_neck': args.mogev2bb_use_neck
                }
            }

            neck_cfg = getattr(args, 'mogev2bb_neck_config_override', None)
            if neck_cfg is not None:
                moge_args['neck_config_override'] = neck_cfg

            self.backbone = MoGeV2BB.from_pretrained(**moge_args)
            # TODO: temporary workaround
            self.backbone.encoder.backbone.mask_token.requires_grad = False

        def forward(self, images) -> List[torch.Tensor]:
            features, _ = self.backbone(images)
            return features

    return Adapter()
