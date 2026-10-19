"""Download the pinned dataset and split complete songs before making windows."""
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def load_data():
    source = json.loads((ROOT/'data_source.json').read_text())
    path = ROOT/'data/irish.abc'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(source['url'], timeout=60).read())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256'], 'Dataset hash changed'
    text = raw.decode('utf-8').replace('\r\n', '\n')
    # Blank lines also occur inside some tunes; X: marks the actual tune boundary.
    songs = [s.strip() for s in re.split(r'(?m)(?=^X:)', text) if s.strip()]
    assert all(s.startswith('X:') for s in songs)
    # Ignore IDs and titles when grouping duplicates. Keep repeated bodies together.
    def body(song):
        return '\n'.join(line.strip() for line in song.splitlines()
                         if not line.startswith(('X:', 'T:', 'Z:')))
    groups = {}
    for song in songs:
        groups.setdefault(body(song), []).append(song)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    count = max(1, round(0.1*len(keys)))
    valid_keys = set(keys[:count])
    train = [s+'\n\n' for k in keys[count:] for s in groups[k]]
    valid = [s+'\n\n' for k in keys[:count] for s in groups[k]]
    assert not {body(s.strip()) for s in train} & {body(s.strip()) for s in valid}
    # ID zero is reserved for characters not seen in training.
    chars = ['<UNK>'] + sorted(set(''.join(train)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([char_to_id.get(c, 0) for c in s], dtype=torch.long)
    details = dict(songs=len(songs),groups=len(groups),train_songs=len(train),
        validation_songs=len(valid),vocab_size=len(chars),seed=42,
        validation_unknown_chars=sum(c not in char_to_id for s in valid for c in s),
        train_song_hashes=[hashlib.sha256(s.encode()).hexdigest() for s in train],
        validation_song_hashes=[hashlib.sha256(s.encode()).hexdigest() for s in valid])
    return [encode(s) for s in train], [encode(s) for s in valid], chars, details


def make_windows(songs, length):
    # The stride is fixed at 64 for both context-length settings. No window crosses songs.
    windows = [song[start:start+length+1] for song in songs
               for start in range(0, len(song)-length, 64)]
    if not windows:
        raise ValueError('No songs are long enough for this context length')
    return torch.stack(windows)
