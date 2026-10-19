"""Training helpers shared by all six runs."""
import math
import random
import time
import torch
import torch.nn.functional as F
from models import build_model
from data import make_windows


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, windows, device):
    model.eval()
    total, tokens = 0.0, 0
    for batch in windows.split(32):
        batch = batch.to(device)
        logits = model(batch[:, :-1])
        target = batch[:, 1:]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 target.reshape(-1), reduction='sum').item()
        tokens += target.numel()
    return total/tokens


def train_one(config, training_songs, validation_windows, chars, folder, device):
    set_seed(config['seed'])
    model = build_model(config['model'], len(chars), config['width']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    windows = make_windows(training_songs, config['length']).to(device)
    generator = torch.Generator().manual_seed(config['seed'])
    steps = config['token_budget']//(config['batch_size']*config['length'])
    history, train_seconds = [], 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(1, steps+1):
        indices = torch.randint(len(windows), (config['batch_size'],), generator=generator)
        batch = windows[indices.to(device)]
        model.train()
        if device.type == 'cuda': torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        train_seconds += time.perf_counter()-tick
        if step % (steps//4) == 0:
            valid_loss = evaluate(model, validation_windows, device)
            row = dict(step=step,seen_tokens=step*config['batch_size']*config['length'],
                       train_loss=loss.item(),validation_loss=valid_loss)
            history.append(row)
            print(config['name'], row, flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state=model.state_dict(),config=config,chars=chars),folder/'model.pt')
    return dict(config=config,parameters=sum(p.numel() for p in model.parameters()),
        steps=steps,validation_loss=history[-1]['validation_loss'],
        validation_perplexity=math.exp(history[-1]['validation_loss']),history=history,
        training_seconds=train_seconds,total_seconds=time.perf_counter()-start,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/1e6 if device.type=='cuda' else None,
        training_windows=len(windows),validation_windows=len(validation_windows))
