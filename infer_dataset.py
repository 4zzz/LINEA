import argparse
import os
from pathlib import Path
import subprocess
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import build_dataset, BatchImageCollateFunction
import numpy as np
from PIL import Image, ImageDraw
from util.prediction_record import PredictionRecord, make_prediction_file, save, utc_timestamp
from models.linea.matcher import build_matcher
from models.linea.moge.utils.alignment import align_points_scale_xyz_shift, align_points_scale_z_shift

def draw(images, lines, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        line = lines[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, l in enumerate(line):
            draw.line(list(l), fill="red", width=5)
            draw.text(
                (l[0], l[1]),
                text=f"{round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images

#if __name__ == '__main__':
parser = argparse.ArgumentParser(
    'Produce inference files using trained model for all dataset samples',
    #parents=[get_args_parser(all_optional=True)],
)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--split', type=str, choices=('test', 'val', 'train'), default='test')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--model', type=str)
parser.add_argument('--save_png_visualization', action='store_true', default=False)
parser.add_argument('-d', '--dont_save_sample', action='store_true', default=False)
parser.add_argument('-o', '--output_directory', type=str)
parser.add_argument('--pred_threshold', type=float, default=0.0)
parser.add_argument('--max_samples', type=int, default=None)
parser.add_argument('--single_file', type=str, default=None,
                    help='Save all prediction records into one JSON/JSON.GZ file instead of one file per sample.')
parser.add_argument('--fit-affine', action='store_true', default=False,
                    help='Fit training-style affine scale/shift from matched 2D lines and save prediction.lines3d_fitted.')

args = parser.parse_args()

checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
model_args = checkpoint['args']
if not hasattr(model_args, 'linea3d'):
    model_args.linea3d = False

#for name, value in vars(args).items():
#    if name in model_args:
#        print(f"Overriding value of '{name}' in model config from {getattr(model_args, name)} to {value}")
#        setattr(model_args, name, value)

def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)

# build model
model, postprocessor = create(model_args, 'modelname')

if "ema" in checkpoint:
    state = checkpoint["ema"]["module"]
else:
    state = checkpoint["model"]
model.load_state_dict(state)

device = args.device
if device.startswith('cuda') and not torch.cuda.is_available():
    print("CUDA requested but not available, falling back to CPU.")
    device = 'cpu'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        raw_outputs = self.model(images)
        lines, scores = self.postprocessor(raw_outputs, orig_target_sizes)
        return raw_outputs, lines, scores

dataset = build_dataset(image_set=args.split, args=model_args)

if model_args.eval_spatial_size is not None:
    collate_fn = BatchImageCollateFunction(base_size=model_args.eval_spatial_size[0])
else:
    collate_fn = BatchImageCollateFunction()

dataloader = DataLoader(
    dataset,
    args.batch_size,
    #sampler=sampler_val,
    drop_last=False,
    collate_fn=collate_fn,
    #num_workers=args.num_workers
)

model = Model().to(device)
model.eval()

matcher = build_matcher(model_args) if args.fit_affine else None

output_directory = Path(args.output_directory) if args.output_directory is not None else None
if output_directory is None and args.single_file is None:
    parser.error('--output_directory is required unless --single_file is set.')
if args.save_png_visualization and output_directory is None:
    parser.error('--output_directory is required when --save_png_visualization is set.')

if output_directory is not None:
    output_directory.mkdir(parents=True, exist_ok=True)

single_file_path = None
if args.single_file is not None:
    single_file_path = Path(args.single_file)
    if not single_file_path.is_absolute() and output_directory is not None:
        single_file_path = output_directory / single_file_path


def _git_output(*cmd):
    try:
        return subprocess.check_output(cmd, cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:
        return None


def _move_targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _align_points(src_pts, tgt_pts, alignment):
    src_pts_flat = src_pts.reshape(1, -1, 3)
    tgt_pts_flat = tgt_pts.reshape(1, -1, 3)
    weight = torch.ones(src_pts_flat.shape[:2], dtype=src_pts_flat.dtype, device=src_pts_flat.device)

    if alignment == 'xyz_shift':
        scale, shift = align_points_scale_xyz_shift(src_pts_flat, tgt_pts_flat, weight)
    elif alignment == 'z_shift':
        scale, shift = align_points_scale_z_shift(src_pts_flat, tgt_pts_flat, weight)
    else:
        raise ValueError(f"Unknown line3d_alignment value '{alignment}'.")

    aligned = scale[:, None, None] * src_pts_flat + shift[:, None, :]
    loss = torch.nn.functional.l1_loss(aligned, tgt_pts_flat, reduction='none').sum()
    return scale.reshape(()), shift.reshape(3), loss


def _fit_affine_lines3d(raw_outputs, targets, indices, alignment):
    if 'pred_lines3d' not in raw_outputs:
        return [None for _ in targets]

    fitted = []
    for batch_i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
        sample_lines3d = raw_outputs['pred_lines3d'][batch_i]
        if len(src_idx) == 0 or 'lines3d' not in target:
            fitted.append(None)
            continue

        src_idx = src_idx.to(sample_lines3d.device)
        tgt_idx = tgt_idx.to(target['lines3d'].device)
        src_pts = sample_lines3d[src_idx].view(-1, 2, 3)
        tgt_pts = target['lines3d'][tgt_idx].view(-1, 2, 3)

        direct_scale, direct_shift, direct_loss = _align_points(src_pts, tgt_pts, alignment)
        swap_scale, swap_shift, swap_loss = _align_points(src_pts, tgt_pts[:, [1, 0], :], alignment)
        scale, shift = (swap_scale, swap_shift) if swap_loss < direct_loss else (direct_scale, direct_shift)

        fitted.append((scale * sample_lines3d.view(-1, 2, 3) + shift).view(-1, 6))

    return fitted


def _build_predictions(lines2d, scores, raw_outputs, idx, threshold, fitted_lines3d=None):
    sample_lines2d = lines2d[idx]
    sample_scores = scores[idx]
    keep = sample_scores > threshold

    pred = {
        'lines2d': [
            {
                'endpoints': sample_lines2d[j],
                'score': sample_scores[j],
            }
            for j in range(len(sample_lines2d))
            if keep[j]
        ],
    }

    if 'pred_lines3d' in raw_outputs:
        sample_lines3d = raw_outputs['pred_lines3d'][idx]
        pred['lines3d'] = [
            {
                'endpoints': sample_lines3d[j],
                'score': sample_scores[j],
            }
            for j in range(len(sample_lines3d))
            if keep[j]
        ]

    if fitted_lines3d is not None and fitted_lines3d[idx] is not None:
        sample_lines3d_fitted = fitted_lines3d[idx]
        pred['lines3d_fitted'] = [
            {
                'endpoints': sample_lines3d_fitted[j],
                'score': sample_scores[j],
            }
            for j in range(len(sample_lines3d_fitted))
            if keep[j]
        ]

    return pred


def _raw_output_for_record(raw_outputs, idx):
    sample_output = {}
    for key, value in raw_outputs.items():
        if 'aux' in key or key == 'dn_meta':
            continue
        if isinstance(value, torch.Tensor):
            sample_output[key] = value[idx]
        else:
            sample_output[key] = value
    return sample_output


file_meta = {
    'codebase': {
        'commit': _git_output('git', 'rev-parse', 'HEAD'),
        'diff': _git_output('git', 'diff', 'HEAD'),
    },
    'export': {
        'created_at_utc': utc_timestamp(),
        'weights_path': os.path.abspath(args.model),
        'split': args.split,
        'fit_affine': args.fit_affine,
        'line3d_alignment': getattr(model_args, 'line3d_alignment', None),
    }
}

i = 0
single_file_records = []
with torch.no_grad():
    for samples, targets in dataloader:
        if args.max_samples is not None and i >= args.max_samples:
            break

        orig_target_sizes = np.array([[tgt['orig_size'][1].item(), tgt['orig_size'][0].item()] for tgt in targets])

        targets_device = _move_targets_to_device(targets, device)
        raw_outputs, lines, scores = model(samples.to(device), torch.tensor(orig_target_sizes).to(device))
        if args.fit_affine:
            indices = matcher(raw_outputs, targets_device)
            fitted_lines3d = _fit_affine_lines3d(
                raw_outputs,
                targets_device,
                indices,
                getattr(model_args, 'line3d_alignment', 'xyz_shift'),
            )
        else:
            fitted_lines3d = None

        pil_imgs = [Image.open(tgt['image_path']).convert("RGB") for tgt in targets]
        vis = draw(pil_imgs, lines, scores, thrh=args.pred_threshold)

        for idx in range(len(targets)):
            if args.max_samples is not None and i >= args.max_samples:
                break

            record = PredictionRecord(
                record_id=f"{args.split}_{i:06d}",
                raw_data={
                    'input': {} if args.dont_save_sample else samples[idx],
                    'target': targets[idx],
                    'output_raw': _raw_output_for_record(raw_outputs, idx),
                },
                losses={},
                prediction=_build_predictions(lines, scores, raw_outputs, idx, args.pred_threshold, fitted_lines3d),
                meta={
                    'dataset': {
                        'split': args.split,
                        'image_path': targets[idx].get('image_path'),
                        'image_id': targets[idx].get('image_id'),
                        'orig_size': targets[idx].get('orig_size'),
                    },
                    'model': {
                        'weights_path': os.path.abspath(args.model),
                    },
                    'runtime': {
                        'device': args.device,
                    },
                },
            )
            if single_file_path is not None:
                single_file_records.append(record)
                print('adding prediction data', i)
            else:
                prediction_file = make_prediction_file(
                    dataset_name=model_args.dataset_name,
                    codebase_name='LINEA',
                    records=[record],
                    meta=file_meta,
                )

                prediction_path = output_directory / f'eval_{i:03}.json.gz'
                save(prediction_file, prediction_path)
                print('saving prediction data to', prediction_path)

            if args.save_png_visualization:
                png_path = output_directory / f'eval_{i:03}.png'
                vis[idx].save(png_path)

            i += 1

if single_file_path is not None:
    prediction_file = make_prediction_file(
        dataset_name=model_args.dataset_name,
        codebase_name='LINEA',
        records=single_file_records,
        meta=file_meta,
    )
    save(prediction_file, single_file_path)
    print('saving prediction data to', single_file_path)
