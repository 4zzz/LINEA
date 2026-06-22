from .coco import build as build_coco
from .monolines3d import build_mono3d_from_args
from .line_eval import LineEvaluator
from .collate import BatchImageCollateFunction

def build_dataset(image_set, args):
    dataset_name = getattr(args, 'dataset_name', 'coco')
    if dataset_name == 'coco':
        return build_coco(image_set, args)
    if dataset_name == 'monolines3d':
        return build_mono3d_from_args(image_set, args)
    raise ValueError(f'unknown dataset_name: {dataset_name}')
