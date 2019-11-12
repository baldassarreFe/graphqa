import yaml
import pandas as pd

from pathlib import Path

df = []

test_folder = Path('../../results/allfeatures')
for d in test_folder.iterdir():
    if not d.joinpath('test.yaml').exists():
        continue
    with d.joinpath('test.yaml').open() as f:
        results = yaml.safe_load(f)['test']['metric']
    for output_type, metrics in results.items():
        output_type = output_type.replace('local_', 'Local ').replace('global_', 'Global ') \
            .replace('lddt', 'LDDT').replace('gdtts', 'GDT_TS')
        for metric_name, value in metrics.items():
            if 'pearson' in metric_name or 'kendall' in metric_name or 'spearman' in metric_name:
                metric_name = metric_name.replace('pearson', 'R').replace('kendall', 'τ').replace('spearman', 'ρ')
                if metric_name.startswith('per_'):
                    metric_name = metric_name.rsplit('_', maxsplit=1)
                    metric_name = metric_name[1] + ' ' + metric_name[0].replace('_', ' ')
            elif 'first_rank_loss' == metric_name:
                metric_name = 'First Rank Loss'
            elif 'rmse' == metric_name:
                metric_name = 'RMSE'
            else:
                print('Unrecognized', metric_name)
                continue
            df.append({
                'AsReportedBy': 'Ours',
                'Method': 'GraphQA',
                'TrainValSets': 'CASP 7+8+9+10',
                'TestSet': d.name.replace('CASP', 'CASP ').replace('_stage', ' stage '),
                'Output': output_type,
                'Metric': metric_name,
                'Value': value
            })
            
test_folder = Path('../../results/residueonly')
for d in test_folder.iterdir():
    if not d.joinpath('test.yaml').exists():
        continue
    with d.joinpath('test.yaml').open() as f:
        results = yaml.safe_load(f)['test']['metric']
    for output_type, metrics in results.items():
        output_type = output_type.replace('local_', 'Local ').replace('global_', 'Global ') \
            .replace('lddt', 'LDDT').replace('gdtts', 'GDT_TS')
        for metric_name, value in metrics.items():
            if 'pearson' in metric_name or 'kendall' in metric_name or 'spearman' in metric_name:
                metric_name = metric_name.replace('pearson', 'R').replace('kendall', 'τ').replace('spearman', 'ρ')
                if metric_name.startswith('per_'):
                    metric_name = metric_name.rsplit('_', maxsplit=1)
                    metric_name = metric_name[1] + ' ' + metric_name[0].replace('_', ' ')
            elif 'first_rank_loss' == metric_name:
                metric_name = 'First Rank Loss'
            elif 'rmse' == metric_name:
                metric_name = 'RMSE'
            else:
                print('Unrecognized', metric_name)
                continue
            df.append({
                'AsReportedBy': 'Ours',
                'Method': 'GraphQA-RAW',
                'TrainValSets': 'CASP 7+8+9+10',
                'TestSet': d.name.replace('CASP', 'CASP ').replace('_stage', ' stage '),
                'Output': output_type,
                'Metric': metric_name,
                'Value': value
            })

df = pd.DataFrame(df).sort_values(['AsReportedBy', 'Method', 'TestSet', 'Output', 'Metric'])
df.to_csv('our_results.csv', header=True, index=False)
