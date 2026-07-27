
# Basic usage

Continue to 300 total epochs using checkpoint.pth:
```
tools/continue_training.py output/my_experiment --epochs 300
```

Select a checkpoint:
```
tools/continue_training.py output/my_experiment \
    --checkpoint checkpoint0049.pth \
    --epochs 300
```

Fork with changed settings:
```
tools/continue_training.py output/my_experiment \
    --output-dir output/my_experiment_weight_05 \
    --epochs 300 \
    --set line3d_loss_weight_end=0.5
```

Validate without starting training:
```
tools/continue_training.py output/my_experiment --epochs 300 --dry-run
```

The script automatically:
- Finds the recorded config and latest checkpoint.
- Replays previous --options when continuing an existing fork.
- Permits only epoch extension in the same output directory.
- Requires a new empty directory for semantic changes.
- Rejects architecture changes until --weights-only exists.
- Rejects unsafe use_warmup=True continuation because main.py does not restore that scheduler yet.
- Archives source metadata and writes continuation status/provenance.
- Preserves runtime settings such as AMP, seed and worker count.
