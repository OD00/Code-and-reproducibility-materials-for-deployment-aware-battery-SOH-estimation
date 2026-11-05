"""Configure and build the four frozen Pico 2 firmware targets in Release."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
TARGETS=['paper_exp3_cnnlstmsimplecnn_fused','paper_exp3_cnnmambasimplecnn','paper_exp3_cnntransformer','paper_exp3_resnet1d']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True,help='Existing Pico SDK checkout with initialized submodules')
    p.add_argument('--toolchain',type=Path,help='Directory containing arm-none-eabi-gcc')
    p.add_argument('--picotool-dir',type=Path,help='Directory containing picotoolConfig.cmake; optional offline hint')
    p.add_argument('--build-dir',type=Path,help='Defaults to outputs/firmware-release; uses a temp path on Windows with a non-ASCII repository path')
    p.add_argument('--targets',nargs='+',choices=TARGETS,default=TARGETS)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--generator',default='Ninja')
    p.add_argument('--configure-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    src=ROOT/'firmware/pico_tflmicro'
    default_build=ROOT/'outputs/firmware-release'
    if os.name=='nt' and not str(ROOT).isascii():
        key=hashlib.sha256(str(ROOT).encode('utf-8')).hexdigest()[:12]
        default_build=Path(tempfile.gettempdir())/'battery-soh-builds'/key/'release'
    build=(a.build_dir or default_build).resolve()
    print('Firmware output directory:',build,flush=True)
    if not a.dry_run:
        if not (a.sdk/'pico_sdk_init.cmake').is_file(): raise FileNotFoundError(a.sdk/'pico_sdk_init.cmake')
        if not shutil.which('cmake'): raise RuntimeError('cmake is required on PATH')
    commands=[['cmake','-S',str(src),'-B',str(build),'-G',a.generator,
               '-DCMAKE_BUILD_TYPE=Release','-DPICO_BOARD=pico2','-DPICO_PLATFORM=rp2350-arm-s',
               '-DPICO_SDK_PATH='+str(a.sdk.resolve())]]
    if a.toolchain: commands[0].append('-DPICO_TOOLCHAIN_PATH='+str(a.toolchain.resolve()))
    if a.picotool_dir: commands[0].append('-Dpicotool_DIR='+str(a.picotool_dir.resolve()))
    if not a.configure_only: commands.append(['cmake','--build',str(build),'--target',*a.targets,'--parallel',str(a.jobs)])
    for command in commands:
        print(subprocess.list2cmdline(command),flush=True)
        if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__': main()
