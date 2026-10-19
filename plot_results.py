import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    results=json.loads((args.runs/'summary.json').read_text())
    figures=args.runs/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in results:
        history=r['history'];label=r['config']['name']
        axes[0].plot([h['seen_tokens']/1e6 for h in history],
                     [h['validation_loss'] for h in history],marker='o',label=label)
    axes[0].set(xlabel='Training characters (millions)',ylabel='Validation cross-entropy',
                title='Same 64-character validation windows')
    axes[0].legend(fontsize=7)
    axes[1].barh([r['config']['name'] for r in results],[r['training_seconds'] for r in results])
    axes[1].set(xlabel='Training seconds (evaluation excluded)',title='Local GPU timing')
    fig.tight_layout();fig.savefig(figures/'comparison.png',dpi=170);plt.close(fig)
    if not (args.runs/'sampling_results.json').exists():return
    samples=json.loads((args.runs/'sampling_results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind in ['lstm','transformer']:
        ratios=[];parsed=[]
        for t in [0.7,1.0,1.3]:
            group=[s for s in samples if s['model']==kind and s['temperature']==t]
            ratios.append(sum(s['unique_4gram_ratio'] for s in group)/len(group))
            parsed.append(sum(s['parse_success'] for s in group))
        axes[0].plot([0.7,1.0,1.3],ratios,'o-',label=kind)
        axes[1].plot([0.7,1.0,1.3],parsed,'o-',label=kind)
    axes[0].set(ylabel='Unique character 4-gram ratio',title='Text diversity, not musical quality',ylim=(0,1))
    axes[1].set(ylabel='Samples parsed and exported to MIDI',title='Three samples per temperature',yticks=[0,1,2,3],ylim=(-.1,3.1))
    for ax in axes:ax.set_xlabel('Sampling temperature');ax.legend()
    fig.tight_layout();fig.savefig(figures/'temperature.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
