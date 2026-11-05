"""Prepare the fixed CALCE fold, convert to INT8, and generate firmware assets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PREFIX='paper_experiments.experiment_3_mcu_deployment.'
MODELS=['CnnLSTMSimpleCNN','CnnMambaSimpleCNN','CnnTransformer','ResNet1d']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','prepare','convert','assets'],default='all')
    p.add_argument('--checkpoint-root',type=Path,default=ROOT/'artifacts/checkpoints')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/model_input_a1_v2')
    p.add_argument('--prepared-dir',type=Path,default=ROOT/'outputs/deployment/seed_42_test_CS2_38')
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    prepared=a.prepared_dir.resolve()
    def run(module,*args):
        command=[sys.executable,'-m',PREFIX+module,*map(str,args)]
        print(subprocess.list2cmdline(command),flush=True)
        if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)
    if a.stage in ('all','prepare'):
        run('prepare_inputs','--experiment1-root',a.checkpoint_root.resolve(),'--data-root',a.data_root.resolve(),
            '--models',*a.models,'--seed','42','--test-cell','CS2_38','--test-samples','0',
            '--representative-samples-per-cell','128','--output-dir',prepared)
    for model in a.models:
        fused=prepared/'models'/model/'fused'
        if a.stage in ('all','convert'):
            if model=='CnnLSTMSimpleCNN':
                run('optimization_tests.run_fused_lstm','--prepared-dir',prepared,'--model',model,'--output-dir',fused)
            else:
                run('export_tensorflow_manual','--prepared-dir',prepared,'--model',model)
                run('convert_tensorflow_int8','--prepared-dir',prepared,'--model',model)
        if a.stage in ('all','assets'):
            extra=['--model-dir',str(fused)] if model=='CnnLSTMSimpleCNN' else []
            run('generate_firmware_assets','--prepared-dir',prepared,'--model',model,'--fixture-samples','64',*extra)

if __name__=='__main__': main()
