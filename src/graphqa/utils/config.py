import random
from argparse import Namespace
from typing import Sequence, Mapping, Tuple, Any, Iterator

import namesgenerator
from omegaconf import OmegaConf


def random_name():
    return f"{namesgenerator.get_random_name()}_{random.randint(1000, 9999)}"


def parse_config(arg_list: Sequence[str]):
    OmegaConf.register_resolver("random_seed", lambda: random.randint(0, 10_000))
    OmegaConf.register_resolver("random_name", random_name)

    conf = OmegaConf.create()
    for s in arg_list:
        if s.endswith(".yaml"):
            conf.merge_with(OmegaConf.load(s))
        else:
            conf.merge_with_dotlist([s])

    # Make sure everything is resolved
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    return conf


def flatten_dict(input: Mapping, prefix: Sequence = ()) -> Iterator[Tuple[Tuple, Any]]:
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


def omegaconf_to_namespace(config: OmegaConf) -> Namespace:
    return Namespace(
        **{
            ".".join(k): v
            for k, v in flatten_dict(OmegaConf.to_container(config, resolve=True))
        }
    )


def namespace_to_omegaconf(args: Namespace) -> OmegaConf:
    config = OmegaConf.create()
    for k, v in args:
        config.merge_with_dotlist([f"{k}={v}"])
    return config


def main():
    import sys

    conf = parse_config(sys.argv[1:])
    print(conf.pretty())


if __name__ == "__main__":
    main()
