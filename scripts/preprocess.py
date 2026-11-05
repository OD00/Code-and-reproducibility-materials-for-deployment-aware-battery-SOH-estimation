"""Run the collected raw-record -> QC -> 256-point window pipeline."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','initial','dictionaries','qc','windows','calce'],default='all')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    stages={
        'initial': [
            ['initial_preprocess_datasets_v2.py','--dataset',dataset,'--output-root',str(ROOT/'data/initial'/name)]
            for dataset,name in [('calce','CALCE'),('tju','TJU')]],
        'dictionaries': [['build_cycle_dictionaries_v1.py']],
        'qc': [['process_cycle_quality_v2.py']],
        'windows': [['build_model_input_v1.py','--points','256','--window','12']],
        'calce': [['build_model_input_a1_v2.py']],
    }
    # Original preprocessing expects all selected public cells (8 CALCE/66 TJU),
    # including historical groups; training selects the final manuscript subset.
    for stage,commands in stages.items():
        if a.stage not in ('all',stage): continue
        for command in commands:
            argv=[sys.executable,str(ROOT/'preprocessing'/command[0]),*command[1:]]
            print(subprocess.list2cmdline(argv),flush=True)
            if not a.dry_run: subprocess.run(argv,cwd=ROOT,check=True)

if __name__=='__main__': main()
