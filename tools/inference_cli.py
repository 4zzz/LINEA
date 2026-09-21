"""Shared command-line helpers for inference tools."""


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
