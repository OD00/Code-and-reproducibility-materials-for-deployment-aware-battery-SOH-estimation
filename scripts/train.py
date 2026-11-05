"""Portable entry point for the manuscript's CALCE and TJU training protocols."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['CnnLSTMSimpleCNN', 'CnnMambaSimpleCNN', 'CnnTransformer', 'ResNet1d']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['calce', 'tju'])
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='TJU protocol/data check without training')
    a = p.parse_args()
    if a.validate_only and a.experiment != 'tju': p.error('--validate-only is supported for TJU')
    data = (a.data_root or ROOT/'data'/('model_input_a1_v2' if a.experiment=='calce' else 'model_input_v1/TJU')).resolve()
    out = (a.output_root or ROOT/'outputs'/a.experiment).resolve()
    if not a.dry_run and not data.is_dir(): raise FileNotFoundError(data)
    seeds = a.seeds[:1] if a.validate_only else a.seeds
    for seed in seeds:
        dest = out/f'seed_{seed}'
        if dest.exists() and not a.validate_only:
            raise FileExistsError(f'Refusing to overwrite {dest}; choose a new --output-root')
        if a.experiment == 'calce':
            command = [sys.executable, '-m', 'src.A2.run_model_comparison',
                       '--models', *a.models, '--outer-folds',
                       *['CALCE_group_1C__test_'+c for c in ['CS2_35','CS2_36','CS2_37','CS2_38']],
                       '--epochs', '50', '--lr', '0.0005211317875270432', '--batch-size', '64']
        else:
            command = [sys.executable, '-m', 'paper_experiments.experiment_2_rate_finetune.run_rate_finetune',
                       '--models', *a.models, '--subset-seeds', '101','202','303',
                       '--target-sizes', '1','3','6','all', '--source-epochs','80',
                       '--finetune-epochs','40','--batch-size','64','--source-lr','0.001',
                       '--finetune-lr','0.0001','--patience','12']
            if a.validate_only: command += ['--validate-only']
        command += ['--seed', str(seed), '--data-root', str(data), '--output-dir', str(dest)]
        print(subprocess.list2cmdline(command), flush=True)
        if not a.dry_run: subprocess.run(command, cwd=ROOT, check=True)
    if a.experiment == 'tju' and not a.validate_only and set(a.seeds)=={42,123,2026}:
        command=[sys.executable,'-m','paper_experiments.experiment_2_rate_finetune.aggregate_formal_results',str(out)]
        print(subprocess.list2cmdline(command),flush=True)
        if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)
    if a.experiment == 'calce' and set(a.seeds)=={42,123,2026} and set(a.models)==set(MODELS):
        command=[sys.executable,str(ROOT/'scripts/aggregate_calce.py'),'--run-root',str(out)]
        print(subprocess.list2cmdline(command),flush=True)
        if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__': main()
