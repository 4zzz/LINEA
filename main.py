# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_optim_params
from util.slconfig import DictAction, SLConfig
from util.profiler import stats
import util.misc as utils

from datasets import build_dataset, LineEvaluator, BatchImageCollateFunction
from engine import train_one_epoch, evaluate, test

from tensorboardX import SummaryWriter
from warmup import LinearWarmup


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--coco_path', type=str, default='data/wireframe_processed')
    # training parameters
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--prefetch_factor', '--pretech_factor', dest='prefetch_factor', default=None, type=int)
    parser.add_argument('--no_save_checkpoints', action='store_true')
    parser.add_argument('--no_eval_during_train', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--print_freq', default=500, type=int, help='number of distributed processes')

    return parser


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _git_metadata():
    metadata = {}
    commands = {
        'commit': ['git', 'rev-parse', 'HEAD'],
        'branch': ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        'status_short': ['git', 'status', '--short'],
        'diff_head_binary': ['git', 'diff', 'HEAD', '--binary'],
    }
    for key, command in commands.items():
        try:
            metadata[key] = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            metadata[key] = None
    metadata['untracked_files'] = []
    metadata['untracked_diff_binary'] = None
    try:
        untracked_output = subprocess.check_output(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        untracked_files = [line for line in untracked_output.splitlines() if line]
        metadata['untracked_files'] = untracked_files
        patches = []
        skip_prefixes = ('output/', 'logs/')
        max_untracked_diff_bytes = 1_000_000
        for path in untracked_files:
            normalized_path = path.replace(os.sep, '/')
            if normalized_path.startswith(skip_prefixes):
                patches.append(f'# Skipped generated untracked file: {path}\n')
                continue
            try:
                if os.path.getsize(path) > max_untracked_diff_bytes:
                    patches.append(f'# Skipped large untracked file: {path}\n')
                    continue
            except OSError:
                patches.append(f'# Could not stat untracked file: {path}\n')
                continue
            try:
                patch = subprocess.run(
                    ['git', 'diff', '--no-index', '--binary', '/dev/null', path],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).stdout
                if patch:
                    patches.append(patch)
            except Exception:
                patches.append(f'# Could not diff untracked file: {path}\n')
        metadata['untracked_diff_binary'] = ''.join(patches)
    except Exception:
        pass
    return metadata


def _proc_meminfo():
    meminfo_path = Path('/proc/meminfo')
    if not meminfo_path.exists():
        return None
    meminfo = {}
    try:
        with meminfo_path.open() as f:
            for line in f:
                key, value = line.split(':', 1)
                meminfo[key] = value.strip()
    except Exception:
        return None
    return meminfo


def _cuda_devices():
    cuda = {
        'is_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count(),
        'devices': [],
    }
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            cuda['devices'].append({
                'index': index,
                'name': props.name,
                'total_memory_bytes': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count,
            })
        except Exception as exc:
            cuda['devices'].append({
                'index': index,
                'error': str(exc),
            })
    return cuda


def _nvidia_smi():
    try:
        return subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,name,uuid,memory.total,driver_version',
                '--format=csv,noheader',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _selected_environment():
    prefixes = ('SLURM_',)
    names = {
        'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES',
        'LOCAL_RANK',
        'RANK',
        'WORLD_SIZE',
        'MASTER_ADDR',
        'MASTER_PORT',
        'HOSTNAME',
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in names or any(key.startswith(prefix) for prefix in prefixes)
    }


def _machine_metadata():
    return {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'meminfo': _proc_meminfo(),
        'cuda': _cuda_devices(),
        'nvidia_smi': _nvidia_smi(),
        'environment': _selected_environment(),
    }


def save_run_metadata(args):
    if not getattr(args, 'output_dir', None) or not utils.is_main_process():
        return

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmdline = {
        'argv': sys.argv,
        'cwd': os.getcwd(),
    }
    with open(output_dir / 'cmdline.json', 'w') as f:
        json.dump(cmdline, f, indent=2)

    effective_config = {
        'args': _json_safe(vars(args)),
        'effective_seed': args.seed + utils.get_rank() if hasattr(args, 'seed') else None,
        'python': sys.version,
        'torch': {
            'version': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
        },
    }
    with open(output_dir / 'effective_config.json', 'w') as f:
        json.dump(effective_config, f, indent=2, sort_keys=True)

    with open(output_dir / 'codebase.json', 'w') as f:
        json.dump(_git_metadata(), f, indent=2, sort_keys=True)

    with open(output_dir / 'machine.json', 'w') as f:
        json.dump(_machine_metadata(), f, indent=2, sort_keys=True)


def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)


def _dataloader_kwargs(args):
    kwargs = {
        'num_workers': args.num_workers,
    }
    if args.num_workers > 0 and args.prefetch_factor is not None:
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)

    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # setup tensorboar writer
    if not args.eval:
        writer = SummaryWriter(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        if 'HGNetv2' in args.backbone:
            args.pretrained = False

    # setup eval_spatial_size
    if args.eval_spatial_size is not None and isinstance(args.eval_spatial_size, int):
        size = args.eval_spatial_size
        args.eval_spatial_size = [size, size]

    if args.eval_spatial_size is not None and hasattr(args.eval_spatial_size, "__len__") and len(args.eval_spatial_size) == 2:
        assert args.eval_spatial_size[0] == args.eval_spatial_size[1], 'We only support square shapes'
    save_run_metadata(args)
    device = torch.device(args.device)
    dataloader_kwargs = _dataloader_kwargs(args)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, postprocessors = create(args, 'modelname')
    criterion = create(args, 'criterionname')
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dicts = get_optim_params(args.model_parameters, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    if args.eval:
        dataset_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(
            dataset_val,
            64,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=BatchImageCollateFunction(),
            **dataloader_kwargs,
        )
    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        if hasattr(args.eval_spatial_size, '__len__'):
            collate_fn_train = BatchImageCollateFunction(base_size=args.eval_spatial_size[0], base_size_repeat=3)
            collate_fn_val = BatchImageCollateFunction(base_size=args.eval_spatial_size[0])
        else:
            collate_fn_train = BatchImageCollateFunction()
            collate_fn_val = BatchImageCollateFunction()

        data_loader_train = DataLoader(dataset_train, 
                                        args.batch_size_train, 
                                        sampler=sampler_train, 
                                        drop_last=True,
                                        collate_fn=collate_fn_train,
                                        # pin_memory=dataset_train.pin_memory,
                                        **dataloader_kwargs)
        data_loader_val = DataLoader(dataset_val, 
                                        args.batch_size_val, 
                                        sampler=sampler_val, 
                                        drop_last=False,
                                        collate_fn=collate_fn_val,
                                        # pin_memory=dataset_val.pin_memory,
                                        **dataloader_kwargs)

    # setup lr_drop_list
    if isinstance(args.lr_drop_list , int):
        args.lr_drop_list = [args.lr_drop_list]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list, gamma=0.1)
    warmup_scheduler = LinearWarmup(lr_scheduler, args.warmup_iters) if args.use_warmup else None

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)   
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.milestones = Counter(args.lr_drop_list)
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        evaluator = LineEvaluator()
        test_stats = test(model, criterion, postprocessors, evaluator,
                        data_loader_val, device, args.output_dir, args=args)
        return

    #try:
    #    print(stats(model_without_ddp, args))
    #except Exception as exc:
    #    print(f"Profiler skipped: {exc}")

    print("-"*41 + " Start training " + "-"*42)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler, 
            writer=writer, args=args)
        if warmup_scheduler is None or warmup_scheduler.finished():
            lr_scheduler.step()
        else:
            print(warmup_scheduler.last_step)

        if args.output_dir and not args.no_save_checkpoints:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # Periodic numbered checkpoints are optional on top of the rolling latest checkpoint.
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'warmup_scheduler': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights, checkpoint_path)
                
        test_stats = {}
        if not args.no_eval_during_train:
            test_stats = evaluate(
                model, criterion, postprocessors, data_loader_val, device, args.output_dir, args=args
            )

        if utils.is_main_process():
            for k in test_stats:
                writer.add_scalar(f'Test/{k}'.format(k), test_stats[k], epoch)
            
        log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LINEA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
