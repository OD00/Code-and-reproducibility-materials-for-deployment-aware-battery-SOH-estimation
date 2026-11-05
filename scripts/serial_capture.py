"""Capture the USB CDC text protocol emitted by the frozen Pico 2 firmware."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True,help='COM6 on Windows, or /dev/ttyACM0 on Linux')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f'Refusing to overwrite {a.output}')
    command=[sys.executable,'-m','paper_experiments.experiment_3_mcu_deployment.capture_serial_log',
             '--port',a.port,'--baud','115200','--output',str(a.output.resolve()),'--timeout',str(a.timeout)]
    print(subprocess.list2cmdline(command),flush=True)
    if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__': main()
