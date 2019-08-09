from typing import Mapping, Tuple, Any, Optional, MutableMapping, Sequence, Generator, Iterable

import yaml


def flatten_dict(input: Mapping, prefix: Sequence = ()) -> Generator[Tuple[Tuple, Any], None, None]:
    """Flatten a dictionary into a sequence of (tuple_key, value) tuples

    Example:
        >>> for k, v in flatten_dict({'a': 1, 'b': {'x': 2, 'y': 3}}):
        ...     print(k, v)
        ('a',) 1
        ('b', 'x') 2
        ('b', 'y') 3

    """
    for k, v in input.items():
        if isinstance(v, Mapping):
            yield from flatten_dict(v, prefix=(*prefix, k))
        else:
            yield (*prefix, k), v


def build_dict(tuples: Iterable[Tuple[Tuple[str, ...], Any]],
               target: Optional[MutableMapping] = None) -> MutableMapping:
    if target is None:
        target = {}
    for k, v in tuples:
        t = target
        while len(k) > 1:
            kk, *k = k
            t = t.setdefault(kk, {})
        t[k[0]] = v
    return target


def update_rec(target: MutableMapping, source: Mapping):
    for k in source.keys():
        if k in target and isinstance(target[k], Mapping) and isinstance(source[k], Mapping):
            update_rec(target[k], source[k])
        else:
            target[k] = source[k]


def parse_args(args: Optional[Iterable[str]] = None, config: Optional[MutableMapping[str, Any]] = None):
    """Parse a list of configuration strings into a dictionary.

    Args:
        args: List of configuration strings, defaults to ``sys.argv``.
        config: A dictionary to update with the parsed configuration.

    Returns:
        The parsed configuration.

    Example:
        This is an example::

            python main.py \
                config/train.yaml \
                name=test \
                session.epochs=1 \
                session.losses.nodes.weight=4 \
                --session \
                  epochs=3 \
                  batch_size=100 \
                --optimizer \
                  fn=my.module.function \
                --optimizer.kwargs \
                  lr=199 \
                  weight_decay=17 \
                  other=3 \
                -- \
                  comment=hey \
                --model \
                  config/model.yaml \
                --something.different \
                  a=1 \
                  b=2 \
                  c=3
    """
    if args is None:
        import sys
        args = sys.argv[1:]
    if config is None:
        config = {}
    sub_config = config

    for arg in args:
        if arg[:2] == '--':
            sub_config = config
            if len(arg) > 2:
                prefix = arg[2:].split('.')
                while len(prefix) > 0:
                    sub_config = sub_config.setdefault(prefix[0], {})
                    prefix = prefix[1:]
        elif '=' in arg:
            sub_sub_config = sub_config
            name_dotted, value = arg.split('=')
            name_head, *name_rest = name_dotted.split('.')
            while len(name_rest) > 0:
                sub_sub_config = sub_sub_config.setdefault(name_head, {})
                name_head, *name_rest = name_rest
            sub_sub_config[name_head] = yaml.safe_load(value)
        elif arg.rsplit('.', maxsplit=1)[-1] in {'yaml', 'yml'}:
            with open(arg, 'r') as f:
                sub_config_new = yaml.safe_load(f)
                update_rec(sub_config, sub_config_new)

    return config


def main():
    import pyaml

    config = parse_args()

    pyaml.pprint(config, sort_dicts=False)
    print('-' * 80)
    for k, v in flatten_dict(config):
        print(f'{".".join(k)}:\t{v}')


if __name__ == '__main__':
    main()
