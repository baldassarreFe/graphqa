import yaml
import itertools
import subprocess
import collections
from typing import Mapping, Iterable, MutableMapping

import numpy as np
from munch import munchify, AutoMunch


def git_info():
    try:
        import git
        try:
            result = {}
            repo = git.Repo(search_parent_directories=True)
            try:
                result['url'] = repo.remote(name='origin').url
            except ValueError:
                result['url'] = 'git:/' + repo.working_dir
            result['commit'] = repo.head.commit.hexsha
            result['dirty'] = repo.is_dirty()
            if repo.is_dirty():
                # This creates a line-by-line diff, but it's usually too much
                # result['changes'] = [str(diff) for diff in repo.head.commit.diff(other=None, create_patch=True)]
                result['changes'] = []
                for diff in repo.head.commit.diff(other=None):
                    if diff.new_file:
                        result['changes'].append(f'{diff.change_type} {diff.b_path}')
                    elif diff.deleted_file:
                        result['changes'].append(f'{diff.change_type} {diff.a_path}')
                    elif diff.renamed_file:
                        result['changes'].append(f'{diff.change_type} {diff.a_path} -> {diff.b_path}')
                    else:
                        result['changes'].append(f'{diff.change_type} {diff.b_path}')
            if len(repo.untracked_files) > 0:
                # This would list the names of untracked files, which maybe is not desired
                # result['untracked_files'] = repo.untracked_files
                result['untracked_files'] = len(repo.untracked_files)
            return result
        except (git.InvalidGitRepositoryError, ValueError):
            pass
    except ImportError:
        return None


def cuda_info():
    from xml.etree import ElementTree
    try:
        nvidia_smi_xml = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    driver = ''
    devices = {}
    for child in ElementTree.fromstring(nvidia_smi_xml):
        if child.tag == 'driver_version':
            driver = child.text
        elif child.tag == 'gpu':
            devices[f'cuda:{child.find("minor_number").text}'] = {
                'model': child.find('product_name').text,
                'utilization': child.find('utilization').find('gpu_util').text,
                'memory_used': child.find('fb_memory_usage').find('used').text,
                'memory_total': child.find('fb_memory_usage').find('total').text,
            }

    return {'driver': driver, 'devices': devices}


def parse_dotted(string):
    result_dict = {}
    for kv_pair in string.split(' '):
        sub_dict = result_dict
        name_dotted, value = kv_pair.split('=')
        name_head, *name_rest = name_dotted.split('.')
        while len(name_rest) > 0:
            sub_dict = sub_dict.setdefault(name_head, {})
            name_head, *name_rest = name_rest
        sub_dict[name_head] = yaml.safe_load(value)
    return result_dict


def update_rec(target, source):
    for k in source.keys():
        if k in target and isinstance(target[k], Mapping) and isinstance(source[k], Mapping):
            update_rec(target[k], source[k])
        else:
            # AutoMunch should do its job, but sometimes it doesn't
            target[k] = munchify(source[k], AutoMunch)


def import_(fullname):
    import importlib
    package, name = fullname.rsplit('.', maxsplit=1)
    package = importlib.import_module(package)
    return getattr(package, name)


def set_seeds(seed):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def tail(iterable: Iterable, n=1):
    """Return an iterator over the last n items"""
    # tail(3, 'ABCDEFG') --> E F G
    return iter(collections.deque(iterable, maxlen=n))


def sort_dict(mapping: MutableMapping, order: Iterable):
    for key in itertools.chain(filter(mapping.__contains__, order), set(mapping) - set(order)):
        mapping[key] = mapping.pop(key)
    return mapping


class RunningStats(object):
    def __init__(self):
        self._min = +np.inf
        self._max = -np.inf
        self._count = 0
        self.S = 0.0
        self.m = 0.0

    def add(self, x, count=1):
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

        for _ in range(count):
            self._count += 1
            m_prev = self.m
            self.m += (x - self.m) / self._count
            self.S += (x - self.m) * (x - m_prev)

    def add_from(self, xs):
        for x in xs:
            self.add(x)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return self.m

    @property
    def variance(self):
        return self.S / self._count

    @property
    def std(self):
        return np.sqrt(self.variance)

    def __str__(self):
        return f'{self.mean} ± {self.std} (min: {self.min}, max: {self.max}, count: {self.count})'
