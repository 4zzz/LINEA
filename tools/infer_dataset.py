import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from datasets import build_dataset, BatchImageCollateFunction
import numpy as np
from PIL import Image, ImageDraw
from tools.inference_cli import add_argument
from util.git_utils import git_output, git_repository_root
from util.line_model_export import save_line_model_glb
from util.prediction_record import PredictionRecord, make_prediction_file, save, utc_timestamp
from util.slconfig import DictAction
from models.linea.matcher import build_matcher
from models.linea.moge.utils.alignment import align_points_scale_xyz_shift, align_points_scale_z_shift
from models.linea.line3d_alignment import fit_matched_lines3d_alignment

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

parser = argparse.ArgumentParser(
    'Produce inference files using trained model for all dataset samples',
    #parents=[get_args_parser(all_optional=True)],
)
add_argument(parser, '--device', type=str, default='cuda')
add_argument(parser, '--split', type=str, choices=('test', 'val', 'train'), default='test')
add_argument(parser, '--batch-size', type=int, default=1)
add_argument(parser, '--num-workers', type=int, default=1)
add_argument(parser, '--model', type=str)
add_argument(parser, '--save-png-visualization', action='store_true', default=False)
add_argument(parser, '-d', '--dont-save-sample', action='store_true', default=False)
add_argument(parser, '-o', '--output-directory', type=str)
add_argument(parser, '--overwrite', action='store_true', default=False,
             help='Allow existing prediction, model, and visualization files to be replaced.')
add_argument(parser, '--pred-threshold', type=float, default=0.0)
add_argument(parser, '--max-samples', type=int, default=None)
prediction_group = parser.add_mutually_exclusive_group()
add_argument(
    prediction_group,
    '--prediction-files',
    action='store_true',
    help='Save one eval_NNN.json.gz prediction file per dataset sample.',
)
add_argument(
    prediction_group,
    '-p',
    '--single-prediction-file',
    '--single-file',
    type=str,
    default=None,
    help='Save all prediction records into one JSON, JSON.GZ, or HDF5 file.',
)
add_argument(parser, '--glb-models', action='store_true',
             help='Save one eval_NNN.glb 3D line model per dataset sample.')
add_argument(parser, '--model-add-ground-truth', action='store_true',
             help='Include dataset ground-truth lines as a separate colored node in GLB models.')
add_argument(parser, '--model-line-radius', type=float, default=None,
             help='Tube radius used in GLB models; defaults to a fraction of scene extent.')
add_argument(parser, '--fit-affine', action='store_true', default=False,
             help='Fit training-style affine scale/shift from matched 2D lines and use fitted 3D predictions.')
add_argument(parser, '--save-matching', action='store_true', default=False,
             help='Save the exact matcher assignment, cost components, and top alternatives.')
add_argument(parser, '--matching-top-k', type=int, default=5,
             help='Number of lowest-cost prediction alternatives to save for each target line.')
add_argument(parser, '--save-full-matching-cost-matrix', action='store_true', default=False,
             help='Save all matcher cost matrices. This implies --save-matching and can make files large.')
add_argument(parser, '--sample-index', type=int, default=None,
             help='Run inference for one zero-based dataset index.')
add_argument(
    parser,
    '--set_model_args',
    nargs='+',
    action=DictAction,
    default=None,
    metavar='KEY=VALUE',
    help=(
        'Override arguments stored in the checkpoint before constructing the model and dataset. '
        'Multiple values use KEY=VALUE syntax; comma-separated values become lists.'
    ),
)

if __name__ == '__main__':
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model_args = checkpoint['args']
    training_output_dir = getattr(model_args, 'output_dir', None)
    model_args_overrides = args.set_model_args or {}
    for key, value in model_args_overrides.items():
        previous = getattr(model_args, key, '<not set>')
        print(f"Overriding model_args.{key}: {previous!r} -> {value!r}")
        setattr(model_args, key, value)
    if getattr(model_args, 'output_dir', None) is not None:
        print('Clearing model_args.output_dir for inference safety.')
    model_args.output_dir = None
    if not hasattr(model_args, 'linea3d'):
        model_args.linea3d = False
    if not hasattr(model_args, 'line3d_pred_strategy'):
        model_args.line3d_pred_strategy = 'direct'

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

        def forward(self, images, orig_target_sizes, targets=None):
            raw_outputs = self.model(images, targets)
            lines, scores = self.postprocessor(raw_outputs, orig_target_sizes)
            return raw_outputs, lines, scores

    dataset = build_dataset(image_set=args.split, args=model_args, write_dataset_info=False)
    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(dataset):
            parser.error(f'--sample-index must be between 0 and {len(dataset) - 1}.')
        dataset = Subset(dataset, [args.sample_index])
    if args.matching_top_k < 0:
        parser.error('--matching-top-k must be non-negative.')

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
        num_workers=args.num_workers,
    )

    model = Model().to(device)
    model.eval()

    save_matching = args.save_matching or args.save_full_matching_cost_matrix
    matcher = build_matcher(model_args) if args.fit_affine or save_matching else None

    output_directory = Path(args.output_directory) if args.output_directory is not None else None
    uses_output_directory = args.prediction_files or args.glb_models or args.save_png_visualization
    if uses_output_directory and output_directory is None:
        parser.error(
            '--output-directory is required with --prediction-files, --glb-models, '
            'or --save-png-visualization.'
        )
    if not (uses_output_directory or args.single_prediction_file):
        parser.error(
            'Select at least one output: --prediction-files, --single-prediction-file, '
            '--glb-models, or --save-png-visualization.'
        )
    if args.model_add_ground_truth and not args.glb_models:
        parser.error('--model-add-ground-truth requires --glb-models.')
    if (args.save_matching or args.save_full_matching_cost_matrix) and not (
        args.prediction_files or args.single_prediction_file
    ):
        parser.error('--save-matching requires a prediction-file output mode.')
    if args.model_line_radius is not None and args.model_line_radius <= 0:
        parser.error('--model-line-radius must be positive.')
    if args.glb_models and not model_args.linea3d:
        parser.error('--glb-models requires a LINEA3D checkpoint.')

    single_file_path = None
    if args.single_prediction_file is not None:
        single_file_path = Path(args.single_prediction_file)
        if not single_file_path.is_absolute() and output_directory is not None:
            single_file_path = output_directory / single_file_path

    if args.max_samples is not None and args.max_samples < 0:
        parser.error('--max_samples must be non-negative.')

    if args.sample_index is not None:
        planned_dataset_indices = [args.sample_index]
    else:
        planned_dataset_indices = list(range(len(dataset)))
    if args.max_samples is not None:
        planned_dataset_indices = planned_dataset_indices[:args.max_samples]

    planned_output_paths = []
    if single_file_path is not None:
        planned_output_paths.append(single_file_path)
    if args.prediction_files:
        planned_output_paths.extend(
            output_directory / f'eval_{index:03}.json.gz'
            for index in planned_dataset_indices
        )
    if args.glb_models:
        planned_output_paths.extend(
            output_directory / f'eval_{index:03}.glb'
            for index in planned_dataset_indices
        )
    if args.save_png_visualization:
        planned_output_paths.extend(
            output_directory / f'eval_{index:03}.png'
            for index in planned_dataset_indices
        )

    if len(set(planned_output_paths)) != len(planned_output_paths):
        parser.error('Inference output paths collide; choose a different output path.')

    if not args.overwrite:
        existing_paths = [path for path in planned_output_paths if path.exists()]
        if existing_paths:
            preview = ', '.join(str(path) for path in existing_paths[:3])
            if len(existing_paths) > 3:
                preview += f", ... (+{len(existing_paths) - 3} more)"
            parser.error(
                f'Refusing to overwrite existing output files: {preview}. '
                'Pass --overwrite to replace them.'
            )

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)


git_root = git_repository_root(REPO_ROOT)


def _git_output(*cmd):
    return git_output(cmd, git_root)


def _move_targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _fit_line3d_alignment(src_pts, tgt_pts, weight, alignment):
    if alignment == 'xyz_shift':
        return align_points_scale_xyz_shift(src_pts, tgt_pts, weight)
    if alignment == 'z_shift':
        return align_points_scale_z_shift(src_pts, tgt_pts, weight)
    raise ValueError(f"Unknown line3d_alignment value '{alignment}'.")


def _line3d_regression_loss(src_pts, tgt_pts, loss_type, smooth_l1_beta):
    if loss_type == 'l1':
        return torch.nn.functional.l1_loss(src_pts, tgt_pts, reduction='none')
    if loss_type == 'smooth_l1':
        return torch.nn.functional.smooth_l1_loss(
            src_pts,
            tgt_pts,
            reduction='none',
            beta=smooth_l1_beta,
        )
    raise ValueError(f"Unknown line3d_loss_type '{loss_type}'.")


def _fit_affine_lines3d(
    raw_outputs,
    targets,
    indices,
    alignment,
    loss_type,
    smooth_l1_beta,
):
    if 'pred_lines3d' not in raw_outputs:
        empty = [None for _ in targets]
        return empty, empty.copy()

    fitted = []
    alignment_info = []
    for batch_i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
        sample_lines3d = raw_outputs['pred_lines3d'][batch_i]
        if len(src_idx) == 0 or 'lines3d' not in target:
            fitted.append(None)
            alignment_info.append(None)
            continue

        src_idx = src_idx.to(sample_lines3d.device)
        tgt_idx = tgt_idx.to(target['lines3d'].device)
        src_pts = sample_lines3d[src_idx].view(-1, 2, 3)
        tgt_pts = target['lines3d'][tgt_idx].view(-1, 2, 3)

        scale, shift, initial_use_swapped = fit_matched_lines3d_alignment(
            src_pts,
            tgt_pts,
            lambda src, tgt, weight: _fit_line3d_alignment(
                src, tgt, weight, alignment
            ),
            lambda src, tgt: _line3d_regression_loss(
                src, tgt, loss_type, smooth_l1_beta
            ),
        )
        scale = scale.reshape(())
        shift = shift.reshape(3)

        fitted_sample = scale * sample_lines3d.view(-1, 2, 3) + shift
        fitted.append(fitted_sample.view(-1, 6))

        fitted_matched = fitted_sample[src_idx]
        direct_error = _line3d_regression_loss(
            fitted_matched, tgt_pts, loss_type, smooth_l1_beta
        ).flatten(1).mean(-1)
        swapped_error = _line3d_regression_loss(
            fitted_matched, tgt_pts.flip(1), loss_type, smooth_l1_beta
        ).flatten(1).mean(-1)
        final_use_swapped = swapped_error < direct_error
        alignment_info.append({
            'mode': alignment,
            'scale': scale,
            'shift': shift,
            'initial_use_swapped': initial_use_swapped,
            'final_use_swapped': final_use_swapped,
            'aligned_error': torch.minimum(direct_error, swapped_error),
            'fitted_matched_lines3d': fitted_matched.view(-1, 6),
        })

    return fitted, alignment_info


def _cost_values(costs, prediction_index, target_index):
    return {
        name: matrix[prediction_index, target_index]
        for name, matrix in costs.items()
    }


def _build_matching_record(
    matcher,
    costs,
    indices,
    raw_outputs,
    lines2d,
    scores,
    target,
    batch_index,
    top_k,
    include_full_matrix,
    alignment_info=None,
):
    src_indices, tgt_indices = indices
    src_indices = src_indices.to(costs['total'].device)
    tgt_indices = tgt_indices.to(costs['total'].device)
    num_predictions, num_targets = costs['total'].shape

    orig_height, orig_width = target['orig_size'].unbind()
    line_scale = torch.stack([orig_width, orig_height, orig_width, orig_height]).to(
        dtype=target['lines'].dtype
    )
    target_lines2d_pixels = target['lines'] * line_scale

    pairs = []
    matched_by_target = {
        int(target_index): pair_index
        for pair_index, target_index in enumerate(tgt_indices.tolist())
    }
    for pair_index, (prediction_index, target_index) in enumerate(
        zip(src_indices.tolist(), tgt_indices.tolist())
    ):
        ranking = torch.argsort(costs['total'][:, target_index])
        rank = int(torch.nonzero(ranking == prediction_index, as_tuple=False)[0, 0]) + 1
        target_class = int(target['labels'][target_index])
        pair = {
            'prediction_index': prediction_index,
            'target_index': target_index,
            'target_class': target_class,
            'score': scores[batch_index][prediction_index],
            'target_class_probability': raw_outputs['pred_logits'][
                batch_index, prediction_index, target_class
            ].sigmoid(),
            'prediction_rank_for_target': rank,
            'prediction_line2d_normalized': raw_outputs['pred_lines'][batch_index, prediction_index],
            'prediction_line2d_pixels': lines2d[batch_index][prediction_index],
            'target_line2d_normalized': target['lines'][target_index],
            'target_line2d_pixels': target_lines2d_pixels[target_index],
            'cost': _cost_values(costs, prediction_index, target_index),
        }
        if 'pred_lines3d' in raw_outputs and 'lines3d' in target:
            pair['prediction_line3d'] = raw_outputs['pred_lines3d'][batch_index, prediction_index]
            pair['target_line3d'] = target['lines3d'][target_index]
        if alignment_info is not None:
            pair['prediction_line3d_aligned'] = alignment_info['fitted_matched_lines3d'][pair_index]
            pair['endpoint_orientation_for_alignment'] = (
                'swapped' if bool(alignment_info['initial_use_swapped'][pair_index]) else 'direct'
            )
            pair['endpoint_orientation_after_refit'] = (
                'swapped' if bool(alignment_info['final_use_swapped'][pair_index]) else 'direct'
            )
            pair['aligned_line3d_error'] = alignment_info['aligned_error'][pair_index]
        pairs.append(pair)

    alternatives = []
    candidate_count = min(top_k, num_predictions)
    for target_index in range(num_targets):
        candidate_indices = torch.argsort(costs['total'][:, target_index])[:candidate_count]
        candidates = []
        for rank, prediction_index_tensor in enumerate(candidate_indices, start=1):
            prediction_index = int(prediction_index_tensor)
            candidates.append({
                'prediction_index': prediction_index,
                'rank': rank,
                'selected': matched_by_target.get(target_index) is not None
                    and int(src_indices[matched_by_target[target_index]]) == prediction_index,
                'score': scores[batch_index][prediction_index],
                'cost': _cost_values(costs, prediction_index, target_index),
            })
        alternatives.append({
            'target_index': target_index,
            'candidates': candidates,
        })

    matched_predictions = set(src_indices.tolist())
    matched_targets = set(tgt_indices.tolist())
    record = {
        'method': type(matcher).__name__,
        'cost_weights': {
            'classification': matcher.cost_class,
            'line2d': matcher.cost_line,
        },
        'num_predictions': num_predictions,
        'num_targets': num_targets,
        'pairs': pairs,
        'unmatched_prediction_indices': [
            index for index in range(num_predictions) if index not in matched_predictions
        ],
        'unmatched_target_indices': [
            index for index in range(num_targets) if index not in matched_targets
        ],
        'alternatives_by_target': alternatives,
    }
    if alignment_info is not None:
        record['alignment'] = {
            'mode': alignment_info['mode'],
            'scale': alignment_info['scale'],
            'shift': alignment_info['shift'],
            'num_matched_lines': len(src_indices),
        }
    if include_full_matrix:
        record['full_cost_matrix'] = costs
    return record


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


if __name__ == '__main__':
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
            'save_matching': save_matching,
            'matching_top_k': args.matching_top_k,
            'save_full_matching_cost_matrix': args.save_full_matching_cost_matrix,
            'sample_index': args.sample_index,
            'model_args_overrides': model_args_overrides,
            'training_output_dir': training_output_dir,
            'overwrite': args.overwrite,
            'prediction_files': args.prediction_files,
            'single_prediction_file': None if single_file_path is None else str(single_file_path),
            'glb_models': args.glb_models,
            'model_add_ground_truth': args.model_add_ground_truth,
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
            raw_outputs, lines, scores = model(
                samples.to(device),
                torch.tensor(orig_target_sizes).to(device),
                targets_device,
            )
            if matcher is not None:
                matching_costs = matcher.compute_costs(raw_outputs, targets_device)
                indices = matcher.match_from_costs(matching_costs)
            else:
                matching_costs = None
                indices = None

            if indices is not None and (args.fit_affine or save_matching):
                fitted_lines3d, alignment_info = _fit_affine_lines3d(
                    raw_outputs,
                    targets_device,
                    indices,
                    getattr(model_args, 'line3d_alignment', 'xyz_shift'),
                    getattr(model_args, 'line3d_loss_type', 'l1'),
                    getattr(model_args, 'line3d_smooth_l1_beta', 1.0),
                )
            else:
                fitted_lines3d = None
                alignment_info = None

            if args.save_png_visualization:
                pil_imgs = [Image.open(tgt['image_path']).convert("RGB") for tgt in targets]
                vis = draw(pil_imgs, lines, scores, thrh=args.pred_threshold)
            else:
                vis = None

            for idx in range(len(targets)):
                if args.max_samples is not None and i >= args.max_samples:
                    break

                dataset_index = args.sample_index if args.sample_index is not None else i
                sample_matching = {}
                if save_matching:
                    sample_matching = _build_matching_record(
                        matcher=matcher,
                        costs=matching_costs[idx],
                        indices=indices[idx],
                        raw_outputs=raw_outputs,
                        lines2d=lines,
                        scores=scores,
                        target=targets_device[idx],
                        batch_index=idx,
                        top_k=args.matching_top_k,
                        include_full_matrix=args.save_full_matching_cost_matrix,
                        alignment_info=None if alignment_info is None else alignment_info[idx],
                    )

                if args.prediction_files or single_file_path is not None:
                    record_meta = {
                        'dataset': {
                            'split': args.split,
                            'index': dataset_index,
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
                    }
                    if save_matching:
                        record_meta['matching'] = sample_matching

                    record = PredictionRecord(
                        record_id=f"{args.split}_{dataset_index:06d}",
                        raw_data={
                            'input': {} if args.dont_save_sample else samples[idx],
                            'target': targets[idx],
                            'output_raw': _raw_output_for_record(raw_outputs, idx),
                        },
                        losses={},
                        prediction=_build_predictions(
                            lines,
                            scores,
                            raw_outputs,
                            idx,
                            args.pred_threshold,
                            fitted_lines3d if args.fit_affine else None,
                        ),
                        meta=record_meta,
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

                        prediction_path = output_directory / f'eval_{dataset_index:03}.json.gz'
                        save(prediction_file, prediction_path)
                        print('saving prediction data to', prediction_path)

                if args.glb_models:
                    if 'pred_lines3d' not in raw_outputs:
                        raise RuntimeError('LINEA3D output does not contain pred_lines3d.')
                    model_lines = raw_outputs['pred_lines3d'][idx]
                    prediction_name = 'predictions_raw'
                    if args.fit_affine and fitted_lines3d[idx] is not None:
                        model_lines = fitted_lines3d[idx]
                        prediction_name = 'predictions_affine_fitted'
                    keep = scores[idx] > args.pred_threshold
                    ground_truth = (
                        targets_device[idx].get('lines3d')
                        if args.model_add_ground_truth else None
                    )
                    model_path = output_directory / f'eval_{dataset_index:03}.glb'
                    save_line_model_glb(
                        model_path,
                        model_lines[keep],
                        ground_truth,
                        line_radius=args.model_line_radius,
                        prediction_name=prediction_name,
                    )
                    print('saving 3D model to', model_path)

                if args.save_png_visualization:
                    png_path = output_directory / f'eval_{dataset_index:03}.png'
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
