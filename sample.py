"""Generate ABC text at three temperatures and attempt MIDI conversion."""
import argparse
import json
import warnings
from pathlib import Path
import torch
from music21 import converter
from data import ROOT
from models import build_model

PROMPT = 'X:1\nT:Generated tune\nM:4/4\nL:1/8\nK:D\n'


@torch.inference_mode()
def generate(model, chars, length, temperature, seed, device):
    ids={c:i for i,c in enumerate(chars)}
    tokens=[ids[c] for c in PROMPT]
    random=torch.Generator(device=device).manual_seed(seed)
    output=''
    for _ in range(512):
        context=torch.tensor([tokens[-length:]],device=device)
        logits=model(context)[0,-1]/temperature
        logits[0]=-torch.inf  # Do not generate the unknown-character placeholder.
        token=int(torch.multinomial(torch.softmax(logits,dim=-1),1,generator=random))
        tokens.append(token)
        output+=chars[token]
        if '\n\n' in output:break
    return PROMPT+output,output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    folder=args.runs/'samples'
    if folder.exists():raise FileExistsError('Samples already exist')
    folder.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results=[]
    for kind in ['lstm','transformer']:
        name=f'{kind}_w128_s128'
        saved=torch.load(args.runs/name/'model.pt',map_location='cpu',weights_only=True)
        config,chars=saved['config'],saved['chars']
        model=build_model(kind,len(chars),config['width']).to(device)
        model.load_state_dict(saved['model_state']);model.eval()
        for temperature in [0.7,1.0,1.3]:
            for seed in [101,102,103]:
                text,body=generate(model,chars,config['length'],temperature,seed,device)
                stem=f'{kind}_t{temperature}_seed{seed}'
                (folder/(stem+'.abc')).write_text(text,encoding='utf-8')
                parsed=False;notes=0;error=None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        score=converter.parse(text,format='abc')
                        notes=len(score.recurse().notes)
                        if notes:
                            score.write('midi',fp=str(folder/(stem+'.mid')))
                            parsed=True
                except Exception as exc:
                    error=type(exc).__name__
                grams=[body[i:i+4] for i in range(max(0,len(body)-3))]
                results.append(dict(model=kind,temperature=temperature,seed=seed,
                    abc_file=stem+'.abc',midi_file=stem+'.mid' if parsed else None,
                    parse_success=parsed,notes=notes,parse_error=error,
                    generated_characters=len(body),hit_character_limit=len(body)==512,
                    unique_4gram_ratio=len(set(grams))/len(grams) if grams else 0))
                print(stem,'parsed:',parsed,'notes:',notes,flush=True)
                (args.runs/'sampling_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
