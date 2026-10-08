from PIL import Image, ImageDraw
from pathlib import Path

import torch

from util.simple_prediction import save_simple_prediction
from util.line_model_export import save_line_model_glb
from util.git_utils import git_output, git_repository_root
from util.prediction_record import (
    HDF5PredictionWriter, PredictionRecord, make_prediction_file, save, utc_timestamp,
)


def prediction_record_path(path, backend=None, *, per_sample=False):
    """Use a recognized combined-file suffix, or select an explicit backend."""
    name = str(path)
    suffixes = ('.json.gz', '.gzjson', '.json', '.hdf5', '.h5', '.gz')
    if not per_sample:
        for suffix in suffixes:
            if name.lower().endswith(suffix):
                if backend is None:
                    return Path(path)
                name = name[:-len(suffix)]
                break
    backend = backend or 'json.gz'
    if backend not in ('json', 'json.gz', 'h5'):
        raise ValueError(f'Unknown prediction record backend: {backend}')
    return Path(name + '.' + backend)


def build_prediction_file_meta(args, model_args, model_meta, repository_root, *, has_ground_truth=True):
    """Collect export metadata with the Git commit and an optional Git diff."""
    root = git_repository_root(repository_root)
    single_path = None if args.single_prediction_record is None else prediction_record_path(
        args.single_prediction_record, args.prediction_record_backend,
    )
    codebase = {'commit': git_output(['git', 'rev-parse', 'HEAD'], root)}
    if args.prediction_record_include_codebase_diff:
        codebase['diff'] = git_output(['git', 'diff', 'HEAD'], root)
    return {
        'codebase': codebase,
        'export': {
            'created_at_utc': utc_timestamp(),
            'weights_path': str(Path(args.checkpoint).resolve()),
            'split': getattr(args, 'split', 'images'),
            'fit_affine': has_ground_truth and bool(getattr(model_args, 'linea3d', False)),
            'save_matching': args.prediction_record_save_matching or args.prediction_record_save_full_matching_cost_matrix,
            'matching_top_k': args.prediction_record_matching_top_k,
            'save_full_matching_cost_matrix': args.prediction_record_save_full_matching_cost_matrix,
            'sample_index': None, 'model_args_overrides': {},
            'training_output_dir': model_meta.get('training_output_dir'),
            'overwrite': True, 'prediction_files': args.prediction_record,
            'single_prediction_file': None if single_path is None else str(single_path),
            'glb_models': args.glb_model, 'model_add_ground_truth': has_ground_truth and args.glb_model,
            'line3d_alignment': getattr(model_args, 'line3d_alignment', None),
            'dataset_args_overrides': getattr(args, 'dataset_args_overrides', {}),
            'pred_threshold': args.pred_threshold,
            'save_exact_sample': args.prediction_record_save_exact_sample,
            'include_codebase_diff': args.prediction_record_include_codebase_diff,
        },
    }


def build_prediction_record(
    *, raw_outputs, lines2d, scores, target, batch_index, sample_index,
    split, checkpoint, device, sample=None, matching=None,
    lines3d=None, lines3d_fitted=None, line_depths=None, record_meta=None,
):
    """Build a record from one sample's already-filtered lines and scores."""
    output = sample_raw_outputs(raw_outputs, batch_index)
    prediction = build_record_predictions(lines2d, scores, lines3d, lines3d_fitted, line_depths)
    meta = {
        'dataset': {
            'split': split, 'index': sample_index, 'image_path': target.get('image_path'),
            'image_id': target.get('image_id'), 'orig_size': target.get('orig_size'),
        },
        'model': {'weights_path': str(Path(checkpoint).resolve())},
        'runtime': {'device': str(device)},
    }
    if matching is not None:
        meta['matching'] = matching
    if record_meta is not None:
        meta.update(record_meta)
    return PredictionRecord(
        record_id=f'{split}_{sample_index:06d}',
        raw_data={'input': {} if sample is None else sample,
                  'target': target, 'output_raw': output},
        losses={}, prediction=prediction, meta=meta,
    )


def save_prediction_records(path, records, *, dataset_name, meta):
    save(make_prediction_file(dataset_name, 'LINEA', records, meta=meta), path)


class PredictionRecordWriter:
    """Stream HDF5 records or collect CPU records for a combined JSON file."""

    def __init__(self, path, *, dataset_name, meta):
        self.path, self.dataset_name, self.meta = Path(path), dataset_name, meta
        self.records, self.writer = [], None

    def __enter__(self):
        if self.path.suffix.lower() in ('.h5', '.hdf5'):
            self.writer = HDF5PredictionWriter(self.path, self.dataset_name, 'LINEA', meta=self.meta)
        return self

    def append(self, record):
        if self.writer is not None:
            self.writer.append(record)
        else:
            self.records.append(record)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.close()
        elif exc_type is None:
            save_prediction_records(self.path, self.records, dataset_name=self.dataset_name, meta=self.meta)

def draw_2d_lines(image, lines, scores, gt_lines=None):
    """Return an RGB image with pixel-coordinate lines and prediction scores."""
    image = image.convert("RGB")
    drawing = ImageDraw.Draw(image)

    def as_list(values):
        if hasattr(values, "detach"):
            values = values.detach().cpu()
        return values.tolist()

    if gt_lines is not None:
        for endpoints in as_list(gt_lines.reshape(-1, 4)):
            drawing.line(endpoints, fill="green", width=3)

    endpoints_list = as_list(lines.reshape(-1, 4))
    score_list = as_list(scores.reshape(-1))
    for endpoints, score in zip(endpoints_list, score_list):
        drawing.line(endpoints, fill="red", width=5)
        drawing.text((endpoints[0], endpoints[1]), f"{score:.2f}", fill="blue")
    return image


def build_simple_prediction(
    lines2d,
    lines3d,
    lines3d_fitted,
    scores,
):
    prediction = {
        "scores": scores,
        "lines2d": lines2d,
    }
    if lines3d is not None:
        prediction["lines3d"] = lines3d
    if lines3d_fitted is not None:
        prediction["lines3d_fitted"] = lines3d_fitted
    return prediction


def _cost_values(costs, prediction_index, target_index):
    return {
        name: matrix[prediction_index, target_index]
        for name, matrix in costs.items()
    }


def build_matching_record(
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
            if 'fitted_matched_lines3d' in alignment_info:
                pair['prediction_line3d_aligned'] = alignment_info['fitted_matched_lines3d'][pair_index]
            elif 'prediction_line3d' in pair:
                pair['prediction_line3d_aligned'] = (
                    alignment_info['scale'].reshape(()) * pair['prediction_line3d'].reshape(2, 3)
                    + alignment_info['shift'].reshape(3)
                ).reshape(-1)
            if 'initial_use_swapped' in alignment_info:
                pair['endpoint_orientation_for_alignment'] = (
                    'swapped' if bool(alignment_info['initial_use_swapped'][pair_index]) else 'direct'
                )
            if 'final_use_swapped' in alignment_info:
                pair['endpoint_orientation_after_refit'] = (
                    'swapped' if bool(alignment_info['final_use_swapped'][pair_index]) else 'direct'
                )
            if 'aligned_error' in alignment_info:
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


def build_record_predictions(lines2d, scores, lines3d=None, lines3d_fitted=None, line_depths=None):
    """Package filtered arrays in the legacy endpoints/score representation."""
    prediction = {}
    for key, lines in (
        ('lines2d', lines2d), ('lines3d', lines3d), ('lines3d_fitted', lines3d_fitted),
    ):
        if lines is not None:
            prediction[key] = [
                {'endpoints': line, 'score': score}
                for line, score in zip(lines, scores)
            ]
    if line_depths is not None:
        prediction['line_depths'] = [
            {'depths': depths, 'score': score} for depths, score in zip(line_depths, scores)
        ]
    return prediction


def sample_raw_outputs(raw_outputs, idx):
    sample_output = {}
    for key, value in raw_outputs.items():
        if 'aux' in key or key == 'dn_meta':
            continue
        if isinstance(value, torch.Tensor):
            sample_output[key] = value[idx]
        else:
            sample_output[key] = value
    return sample_output


def create_inference_files(
    batch_idx, sample_idx, raw_outputs, lines, scores, targets,
    matching_costs, indices, alignments, *, checkpoint, device, dataset_name,
    base_name_fn=None, output_paths=None,
    pred_threshold=0.0, simple_json=False, glb_model=False, lines_2d_png=False,
    prediction_record=False, single_prediction_record=None,
    prediction_record_backend=None, prediction_record_save_exact_sample=False,
    prediction_record_save_matching=False, prediction_record_matching_top_k=5,
    prediction_record_save_full_matching_cost_matrix=False, split='test',
    line3d_alignment='xyz_shift', samples=None, matcher=None,
    file_meta=None, stack=None, line_depths=None, record_meta=None,
    glb_prediction_name='predictions',
):
    """Write one sample's outputs using explicit export options and model metadata.

    base_name_fn(sample_idx, target) returns a path without export suffixes.
    Omit it or return None for combined-only output.
    output_paths may override paths for simple_json, glb_model, lines_2d_png,
    and prediction_record. line_depths contains unfiltered batched depths.
    """

    if single_prediction_record is not None and stack is None:
        raise ValueError('Combined prediction records require an exit stack')

    base_name = base_name_fn(sample_idx, targets[batch_idx]) if base_name_fn is not None else None
    if base_name is not None:
        base_name = Path(base_name)

    print(f'Creating files for sample {sample_idx}', base_name)

    if base_name is not None:
        base_name.parent.mkdir(parents=True, exist_ok=True)

    def output_path(kind, default):
        path = Path((output_paths or {}).get(kind, default))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    batch_scores = scores
    sample_scores = scores[batch_idx]
    keep = sample_scores > pred_threshold
    lines2d = lines[batch_idx][keep]
    lines3d = raw_outputs["pred_lines3d"][batch_idx][keep] if 'pred_lines3d' in raw_outputs else None
    sample_line_depths = line_depths[batch_idx][keep] if line_depths is not None else None
    alignment = alignments[batch_idx] if alignments is not None else None
    lines3d_fitted = None
    if alignment is not None and lines3d is not None:
        lines3d_fitted = (
            alignment['scale'].reshape(1, 1, 1) * lines3d.reshape(-1, 2, 3)
            + alignment['shift'].reshape(1, 1, 3)
        ).reshape_as(lines3d)
    scores = sample_scores[keep]

    gt_lines3d = targets[batch_idx].get('lines3d')

    if simple_json:
        prediction = build_simple_prediction(
            lines2d=lines2d, 
            lines3d=lines3d, 
            lines3d_fitted=lines3d_fitted,
            scores=scores
        )
        dest = output_path('simple_json', str(base_name) + '_simple.json')
        save_simple_prediction(dest, prediction)

    if glb_model:
        dest = output_path('glb_model', str(base_name) + '.glb')
        save_line_model_glb(
            dest,
            lines3d, lines3d_fitted, gt_lines3d,
            prediction_name=glb_prediction_name,
        )

    if lines_2d_png:
        with Image.open(targets[batch_idx]['image_path']) as image:
            gt_lines2d_pixels = targets[batch_idx].get('original_lines2d')
            visualization = draw_2d_lines(image, lines2d, scores, gt_lines2d_pixels)
        visualization.save(output_path('lines_2d_png', str(base_name) + '.png'))

    if prediction_record or single_prediction_record is not None:
        matching = None
        if prediction_record_save_matching or prediction_record_save_full_matching_cost_matrix:
            matching = build_matching_record(
                matcher, matching_costs[batch_idx], indices[batch_idx], raw_outputs,
                lines, scores=batch_scores,
                target=targets[batch_idx], batch_index=batch_idx,
                top_k=prediction_record_matching_top_k,
                include_full_matrix=prediction_record_save_full_matching_cost_matrix,
                alignment_info=None if alignment is None else {
                    **alignment, 'mode': line3d_alignment,
                },
            )
        if prediction_record_save_exact_sample and samples is None:
            raise ValueError('Exact-sample records require the model input samples')
        record = build_prediction_record(
            raw_outputs=raw_outputs, lines2d=lines2d, scores=scores,
            lines3d=lines3d, lines3d_fitted=lines3d_fitted,
            target=targets[batch_idx], batch_index=batch_idx, sample_index=sample_idx,
            split=split, checkpoint=checkpoint, device=device,
            sample=samples[batch_idx] if prediction_record_save_exact_sample else None,
            matching=matching, line_depths=sample_line_depths, record_meta=record_meta,
        )
        if prediction_record:
            save_prediction_records(
                output_path('prediction_record', prediction_record_path(base_name, prediction_record_backend, per_sample=True)),
                [record], dataset_name=dataset_name, meta=file_meta,
            )
        elif single_prediction_record is not None:
            record_writer = getattr(stack, '_prediction_record_writer', None)
            if record_writer is None:
                record_writer = stack.enter_context(PredictionRecordWriter(
                    prediction_record_path(single_prediction_record, prediction_record_backend),
                    dataset_name=dataset_name, meta=file_meta,
                ))
                stack._prediction_record_writer = record_writer
            record_writer.append(record)
        return record
