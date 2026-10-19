"""Run architecture, width and context-length comparisons with equal token budgets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import platform
from pathlib import Path
import torch
from data import ROOT, load_data, make_windows
from train import train_one


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output folder for a new run')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,valid,chars,split=load_data()
    # Every model is evaluated on these same 64-character windows.
    validation=make_windows(valid,64)
    (args.output/'split.json').write_text(json.dumps(split,indent=2))
    environment=dict(python=platform.python_version(),torch=torch.__version__,device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        cuda=torch.version.cuda,precision='float32')
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2))
    configs=[]
    for kind in ['lstm','transformer']:
        for width,length in [(128,128),(64,128),(128,64)]:
            configs.append(dict(name=f'{kind}_w{width}_s{length}',model=kind,width=width,
                length=length,seed=42,batch_size=16,token_budget=2048000))
    (args.output/'design.json').write_text(json.dumps(configs,indent=2))
    results=[]
    for config in configs:
        result=train_one(config,train,validation,chars,args.output/config['name'],device)
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2))
        if device.type=='cuda': torch.cuda.empty_cache()
    print('All six runs completed.',flush=True)


if __name__=='__main__': main()
