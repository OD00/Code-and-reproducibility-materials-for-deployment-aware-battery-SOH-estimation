"""Cell-equal macro means per seed, followed by mean and sample SD over seeds."""
import argparse
import csv
from pathlib import Path
import statistics

MODELS=['CnnLSTMSimpleCNN','CnnMambaSimpleCNN','CnnTransformer','ResNet1d']
SEEDS=[42,123,2026]
CELLS={'CS2_35','CS2_36','CS2_37','CS2_38'}
METRICS=['R2','MSE','RMSE','MAE','MaxError','Bias','NRMSE_range']

def aggregate(root):
    by_model={name:[] for name in MODELS}
    for seed in SEEDS:
        path=root/f'seed_{seed}'/'A2_model_comparison_fold_metrics.csv'
        with path.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
        for model in MODELS:
            selected=[r for r in rows if r['model']==model and r['condition_group']=='CALCE_group_1C']
            if len(selected)!=4 or {r['test_record'] for r in selected}!=CELLS:
                raise ValueError(f'{path}: expected exactly four final folds for {model}')
            by_model[model].append({key:statistics.mean(float(r[key]) for r in selected) for key in METRICS})
    result=[]
    for model,seeds in by_model.items():
        row={'implementation_name':model,'n_training_seeds':3,'folds_per_seed':4}
        for metric in METRICS:
            values=[s[metric] for s in seeds]
            row[metric+'_mean']=statistics.mean(values)
            row[metric+'_std']=statistics.stdev(values)
        result.append(row)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    output=a.output or a.run_root/'calce_three_seed_aggregate.csv'
    if output.exists(): raise FileExistsError(output)
    rows=aggregate(a.run_root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(output)

if __name__=='__main__': main()
