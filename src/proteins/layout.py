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
        'Local LDDT': {
            'RMSE': ['MultiLine', ['(train|val)/metric/local_lddt/rmse']],
            'Pearson': ['MultiLine', ['(train|val)/metric/local_lddt/pearson$']],
            'Per model Pearson': ['MultiLine', ['(train|val)/metric/local_lddt/per_model_pearson']],
        },
        'Global GDT_TS': {
            'RMSE': ['MultiLine', ['(train|val)/metric/global_gdtts/rmse']],
            'Pearson': ['MultiLine', ['(train|val)/metric/global_gdtts/pearson$']],
            'Per target Pearson': ['MultiLine', ['(train|val)/metric/global_gdtts/per_target_pearson']],
            'First Rank Loss': ['MultiLine', ['(train|val)/metric/global_gdtts/first_rank_loss']],
        },
        
    })

print('Layout saved to', folder)
