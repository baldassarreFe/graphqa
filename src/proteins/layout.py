import argparse
from pathlib import Path

from tensorboardX import SummaryWriter
# Use this when they fix the error with SummaryMetadata.PluginData
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The log folder to place the layout in')
args = parser.parse_args()

folder = (Path(args.folder) / 'layout').expanduser().resolve()
with SummaryWriter(folder) as writer:
    writer.add_custom_scalars({
        'Losses': {
            'Local': ['MultiLine', ['(train|val)/loss/local']],
            'Global': ['MultiLine', ['(train|val)/loss/global']],
        },
        'Metrics Local': {
            'RMSE': ['MultiLine', ['(train|val)/metric/local/rmse']],
            'Correlation': ['MultiLine', ['(train|val)/metric/local/correlation$']],
            'Correlation per Model': ['MultiLine', ['(train|val)/metric/local/correlation_per_model']],
            'R2': ['MultiLine', ['(train|val)/metric/local/r2']],
        },
        'Metrics Global': {
            'RMSE': ['MultiLine', ['(train|val)/metric/global/rmse']],
            'Correlation': ['MultiLine', ['(train|val)/metric/global/correlation$']],
            'Correlation per Target': ['MultiLine', ['(train|val)/metric/global/correlation_per_target']],
            'R2': ['MultiLine', ['(train|val)/metric/global/r2']],
        }
    })

print('Layout saved to', folder)
