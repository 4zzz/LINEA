# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable

import torch
import util.misc as utils


_AMP_DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def get_amp_dtype(args):
    name = getattr(args, 'amp_dtype', 'float16')
    try:
        return _AMP_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown amp_dtype {name!r}; expected one of {sorted(_AMP_DTYPES)}."
        ) from exc


def _move_targets_to_device(targets, device):
    return [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]


def _get_line3d_loss_weight(args, criterion, global_step):
    schedule = getattr(args, 'line3d_loss_weight_schedule', None)
    if schedule is None:
        return None

    if 'loss_line3d' not in criterion.weight_dict:
        return None

    base_weight = getattr(criterion, '_base_line3d_loss_weight', None)
    if base_weight is None:
        base_weight = criterion.weight_dict['loss_line3d']
        criterion._base_line3d_loss_weight = base_weight

    start = getattr(args, 'line3d_loss_weight_start', 0.0)
    end = getattr(args, 'line3d_loss_weight_end', base_weight)

    if schedule == 'constant':
        return end

    if schedule == 'linear_warmup':
        warmup_steps = max(1, int(getattr(args, 'line3d_loss_weight_warmup_steps', 1)))
        alpha = min(max(global_step / warmup_steps, 0.0), 1.0)
        return start + alpha * (end - start)

    raise ValueError(f"Unknown line3d_loss_weight_schedule '{schedule}'.")


def _update_loss_weight_meters(metric_logger, criterion):
    for name, value in criterion.weight_dict.items():
        meter_name = f'{name}_w'
        if meter_name not in metric_logger.meters:
            metric_logger.add_meter(meter_name, utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.update(**{meter_name: value})


def _get_raw_loss_dict(criterion):
    return {
        f'{name}_raw': value
        for name, value in getattr(criterion, 'unscaled_loss_dict', {}).items()
    }


def _collect_line3d_depth_stats(outputs, targets):
    if 'pred_line_depths' not in outputs:
        return {}

    pred_depths = outputs['pred_line_depths']
    stats = {
        'pred_depth_mean': pred_depths.mean(),
        'pred_depth_min': pred_depths.min(),
        'pred_depth_max': pred_depths.max(),
    }

    if targets and all('lines3d' in t for t in targets):
        tgt_z = torch.cat([t['lines3d'].view(-1, 2, 3)[..., 2].reshape(-1) for t in targets], dim=0)
        stats.update({
            'tgt_depth_mean': tgt_z.mean(),
            'tgt_depth_min': tgt_z.min(),
            'tgt_depth_max': tgt_z.max(),
        })

    return stats


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None,
                    lr_scheduler=None, warmup_scheduler=None, args=None, ema_m=None,
                    scaler=None):
    amp_dtype = get_amp_dtype(args)
    if scaler is None:
        scaler = torch.amp.GradScaler(
            str(device),
            enabled=args.amp and amp_dtype == torch.float16,
        )
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device)
        targets = _move_targets_to_device(targets, device)

        global_step = epoch * len(data_loader) + i
        line3d_loss_weight = _get_line3d_loss_weight(args, criterion, global_step)
        if line3d_loss_weight is not None:
            criterion.weight_dict['loss_line3d'] = line3d_loss_weight
        _update_loss_weight_meters(metric_logger, criterion)


        with torch.amp.autocast(str(device), enabled=args.amp, dtype=amp_dtype):
            outputs = model(samples, targets)
        
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        raw_loss_dict_reduced = utils.reduce_dict(_get_raw_loss_dict(criterion))
        depth_stats = _collect_line3d_depth_stats(outputs, targets)
        depth_stats_reduced = utils.reduce_dict(depth_stats) if depth_stats else {}
        losses_reduced_scaled = sum(loss_dict_reduced.values())
        losses_reduced_raw = sum(raw_loss_dict_reduced.values())

        loss_value = losses_reduced_scaled.item()
        raw_loss_value = losses_reduced_raw.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
                    
        if warmup_scheduler is not None:
            warmup_scheduler.step() 

        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(
            loss=loss_value,
            loss_raw=raw_loss_value,
            **loss_dict_reduced,
            **raw_loss_dict_reduced,
        )
        if depth_stats_reduced:
            metric_logger.update(**{k: v.item() for k, v in depth_stats_reduced.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            writer.add_scalar('LossRaw/total', raw_loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            for k, v in raw_loss_dict_reduced.items():
                writer.add_scalar(f'LossRaw/{k.removesuffix("_raw")}', v.item(), global_step)
            for k, v in depth_stats_reduced.items():
                writer.add_scalar(f'Depth/{k}', v.item(), global_step)
            for name, value in criterion.weight_dict.items():
                writer.add_scalar(f'LossWeight/{name}', value, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
   
    header = 'Test:'
    print_freq = args.print_freq
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = _move_targets_to_device(targets, device)

        # Validation can overflow in autocast even when the checkpoint is finite;
        # keep training AMP enabled but compute validation loss in FP32.
        with torch.amp.autocast(str(device), enabled=False):
            outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        raw_loss_dict_reduced = utils.reduce_dict(_get_raw_loss_dict(criterion))
        depth_stats = _collect_line3d_depth_stats(outputs, targets)
        depth_stats_reduced = utils.reduce_dict(depth_stats) if depth_stats else {}
        _update_loss_weight_meters(metric_logger, criterion)
        metric_logger.update(
            loss=sum(loss_dict_reduced.values()),
            loss_raw=sum(raw_loss_dict_reduced.values()),
            **loss_dict_reduced,
            **raw_loss_dict_reduced,
        )
        if depth_stats_reduced:
            metric_logger.update(**{k: v.item() for k, v in depth_stats_reduced.items()})
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats

@torch.no_grad()
def test(model, criterion, postprocessors, evaluator, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    evaluator.cleanup()
   
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 250, header):
        samples = samples.to(device)
        targets = _move_targets_to_device(targets, device)

        outputs = model(samples, targets)

        evaluator.update(outputs, targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.accumulate()
    evaluator.summarize()

            
    return
