"""Shared command-line helpers for inference tools."""


from pathlib import Path


def option_aliases(*option_strings):
    """Return long-option spellings with hyphens and underscores interchangeable."""
    aliases = []
    for option in option_strings:
        variants = [option]
        if option.startswith('--'):
            name = option[2:]
            variants.extend((f"--{name.replace('_', '-')}", f"--{name.replace('-', '_')}"))
        for variant in variants:
            if variant not in aliases:
                aliases.append(variant)
    return aliases


def add_argument(parser_or_group, *option_strings, **kwargs):
    return parser_or_group.add_argument(*option_aliases(*option_strings), **kwargs)

def add_model_args(parser): 
    add_argument(parser, '--device', type=str, default='cuda')
    add_argument(parser, "--checkpoint", required=True, type=Path)

def add_data_loading_args(parser, include_split: bool = True):
    if include_split:
        add_argument(parser, '--split', type=str, choices=('test', 'val', 'train'), default='test')
    add_argument(parser, '--batch-size', type=int, default=1)
    add_argument(parser, '--num-workers', type=int, default=1)

def add_inference_option_args(parser):
    add_argument(
        parser, "--pred-threshold", type=float, default=0.0,
        help="Filter out predictions with lower confidence score than set threshold"
    )

def add_output_args(parser, *, aliases=None):
    """Add common output options, optionally attaching legacy CLI spellings."""
    aliases = aliases or {}

    def add_output_argument(group, option, **kwargs):
        add_argument(group, option, *aliases.get(option, ()), **kwargs)

    add_output_argument(
        parser, "--simple-json", action="store_true",
        help="Save predicted lines in simplified json"
    )
    
    prediction_record_group = parser.add_mutually_exclusive_group()
    add_output_argument(
        prediction_record_group,
        "--prediction-record",
        action="store_true",
        help="Save per sample prediction record",
    )
    add_output_argument(
        prediction_record_group,
        "--single-prediction-record",
        type=Path,
        default=None,
        help="Save single prediction record file with all predictions",
    )
    add_output_argument(
        parser, '--prediction-record-save-exact-sample', action="store_true", default=False,
        help="Save exact sample as it went into model"
    )
    add_output_argument(
        parser, '--prediction-record-backend',
        choices=['json', 'json.gz', 'h5']
    )
    add_output_argument(
        parser, '--prediction-record-include-codebase-diff',
        action='store_true', default=False,
        help='Include the Git diff in prediction-record metadata (default: false).',
    )
    add_output_argument(
        parser, '--prediction-record-save-matching', action='store_true',
        help='Save matcher assignments, costs, and top alternatives in prediction records.',
    )
    add_output_argument(
        parser, '--prediction-record-matching-top-k', type=int, default=5,
        help='Number of lowest-cost alternatives to save for each target line (default: 5).',
    )
    add_output_argument(
        parser, '--prediction-record-save-full-matching-cost-matrix', action='store_true',
        help='Save all matcher cost matrices; implies saving matching information.',
    )
    add_output_argument(
        parser, "--glb-model", action="store_true",
        help="Save 3D line model in GLB format"
    )
    add_output_argument(
        parser, "--lines-2d-png", action="store_true",
        help="Save sample image with visualized 2D lines"
    )
    add_output_argument(
        parser, "--output-dir", type=Path,
        help="Output directory for inference"
    )


def validate_output_args(parser, args):
    """Validate relationships between the shared inference output options."""
    per_sample_output = any((
        args.simple_json,
        args.prediction_record,
        args.glb_model,
        args.lines_2d_png,
    ))
    prediction_record_output = (
        args.prediction_record or args.single_prediction_record is not None
    )

    if not per_sample_output and not prediction_record_output:
        parser.error(
            "Select at least one output: --simple-json, --prediction-record, "
            "--single-prediction-record, --glb-model, or --lines-2d-png."
        )

    if per_sample_output and args.output_dir is None:
        parser.error(
            "--output-dir is required for per-sample outputs: --simple-json, "
            "--prediction-record, --glb-model, and --lines-2d-png."
        )

    if args.prediction_record_backend is not None and not prediction_record_output:
        parser.error(
            "--prediction-record-backend requires --prediction-record or "
            "--single-prediction-record."
        )

    if args.prediction_record_save_exact_sample and not prediction_record_output:
        parser.error(
            "--prediction-record-save-exact-sample requires --prediction-record "
            "or --single-prediction-record."
        )

    if args.prediction_record_matching_top_k < 0:
        parser.error('--prediction-record-matching-top-k must be nonnegative')

    matching_modifiers = (
        ('--prediction-record-save-matching', args.prediction_record_save_matching),
        ('--prediction-record-matching-top-k', args.prediction_record_matching_top_k != 5),
        ('--prediction-record-save-full-matching-cost-matrix',
         args.prediction_record_save_full_matching_cost_matrix),
    )
    for option, enabled in matching_modifiers:
        if enabled and not prediction_record_output:
            parser.error(f'{option} requires --prediction-record or --single-prediction-record.')

    if args.prediction_record_save_full_matching_cost_matrix:
        args.prediction_record_save_matching = True
