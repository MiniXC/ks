import pandas as pd
from glob import glob
from pathlib import Path
import re

def load_synth(src, dest):
    synth_dict = {
        'speaker': [],
        'id': [],
        'wav': [],
        'text': []
    }
    for wav in Path(src).rglob('*.wav'):
        synth_dict['speaker'].append(wav.parent.name)
        synth_dict['id'].append(wav.parent.name + '-' + wav.name.replace('.wav', '').replace('_', '-'))
        synth_dict['wav'].append(str(wav))
        with open(str(wav).replace('.wav', '.lab'), 'r') as lab:
            text = lab.read().upper()
            text = ' '.join(re.findall(r'\w+\'\w+|\w+', text)).strip()
            synth_dict['text'].append(text)
    df = pd.DataFrame(synth_dict)
    # spk2utt
    with open(Path(dest) / 'spk2utt', 'w') as spk2utt:
        for spk in sorted(df['speaker'].unique()):
            utts = df[df['speaker']==spk]['id'].unique()
            utts = sorted([f'{utt}' for utt in utts])
            spk2utt.write(f'{spk} {" ".join(utts)}\n')
    # text
    with open(Path(dest) / 'text', 'w') as text_file:
        for utt in sorted(df['id'].unique()):
            text = df[df['id']==utt]['text'].values[0]
            text_file.write(f'{utt} {text}\n')
    # utt2spk
    with open(Path(dest) / 'utt2spk', 'w') as utt2spk:
        for spk in sorted(df['speaker'].unique()):
            utts = df[df['speaker']==spk]['id'].unique()
            for utt in sorted(utts):
                utt2spk.write(f'{utt} {spk}\n')
    # wav.scp
    with open(Path(dest) / 'wav.scp', 'w') as wavscp:
        for utt in sorted(df['id'].unique()):
            wav = df[df['id']==utt]['wav'].values[0]
            wavscp.write(f'{utt} sox {wav} -r 16000 -c 1 -b 16 -t wav - downsample |\n')