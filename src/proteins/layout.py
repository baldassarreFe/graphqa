import argparse
from pathlib import Path

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The log folder to place the layout in')
args = parser.parse_args()

folder = (Path(args.folder) / 'layout').expanduser().resolve()
with SummaryWriter(folder) as writer:
    writer.add_custom_scalars({
        'Losses': {
            'Local LDDT': ['MultiLine', ['(train|val)/loss/local_lddt']],
            'Global LDDT': ['MultiLine', ['(train|val)/loss/global_lddt']],
            'Global GDT_TS': ['MultiLine', ['(train|val)/loss/global_gdtts']],
        },
        'Metrics Local LDDT': {
            'RMSE': ['MultiLine', ['(train|val)/metric/local_lddt/rmse']],
            'Correlation': ['MultiLine', ['(train|val)/metric/local_lddt/correlation$']],
            'Correlation per Model': ['MultiLine', ['(train|val)/metric/local_lddt/correlation_per_model']],
            'R2': ['MultiLine', ['(train|val)/metric/local_lddt/r2']],
        },
        'Metrics Global LDDT': {
            'RMSE': ['MultiLine', ['(train|val)/metric/global_lddt/rmse']],
            'Correlation': ['MultiLine', ['(train|val)/metric/global_lddt/correlation$']],
            'Correlation per Target': ['MultiLine', ['(train|val)/metric/global_lddt/correlation_per_target']],
            'R2': ['MultiLine', ['(train|val)/metric/global_lddt/r2']],
        },
        'Metrics Global GDT_TS': {
            'RMSE': ['MultiLine', ['(train|val)/metric/global_gdtts/rmse']],
            'Correlation': ['MultiLine', ['(train|val)/metric/global_gdtts/correlation$']],
            'Correlation per Target': ['MultiLine', ['(train|val)/metric/global_gdtts/correlation_per_target']],
            'R2': ['MultiLine', ['(train|val)/metric/global_gdtts/r2']],
        },
        
    })

print('Layout saved to', folder)
