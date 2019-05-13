import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The log folder to place the layout in')
args = parser.parse_args()

folder = (Path(args.folder) / 'layout').expanduser().resolve()
with SummaryWriter(folder) as writer:
    writer.add_custom_scalars({'Losses': {
        'Nodes': ['MultiLine', ['loss/(train|val)/nodes']],
        'Global': ['MultiLine', ['loss/(train|val)/global']],
        'Total': ['MultiLine', ['loss/(train|val)/total']]
    }})

print('Layout saved to', folder)
