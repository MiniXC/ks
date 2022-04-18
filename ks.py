from pathlib import Path
from dataclasses import dataclass, fields
import subprocess
from typing import List
import os
from multiprocessing import cpu_count
import re

from transformers import HfArgumentParser
from rich import print
from rich.console import Console
from rich.progress import track
import wandb
from coolname import generate_slug

from prepare_synth import load_synth

console = Console()
cpus = cpu_count()

@dataclass
class Args:
    wandb_mode: str = 'offline'
    kaldi_path: str = '../kaldi/egs/librispeech/s5'
    data_path: str = '../Data/'
    data_url: str = 'www.openslr.org/resources/12'
    data_name: str = 'LibriSpeech'
    lm_path: str = '{data}/local/lm'
    dict_nosp_path: str = '{data}/local/dict_nosp'
    lang_nosp_path: str = '{data}/lang_nosp'
    lang_nosp_tmp_path: str = '{data}/local/lang_tmp_nosp'
    lm_url: str = 'www.openslr.org/resources/11'
    log_file: str = 'train.log'
    test_sets: str = 'dev_clean,test_clean,dev_other,test_other'
    tts_test_path: str = '../Data/LibriTTS/dev-clean-aligned'
    local_data_path: str = 'data'
    local_exp_path: str = 'exp'
    mfcc_path: str = '{data}/mfcc'
    train_name: str = 'Synthesised'
    train_set: str = 'n_topline'
    synth_test_set: str = 'dev_synth'
    train_cmd: str = 'run.pl'
    lm_names: str = 'tgsmall,tgmed' # tglarge,fglarge
    mfcc_path: str = '{exp}/make_mfcc'
    mono_subset: int = 2000
    mono_path: str = '{exp}/mono'
    tri1_subset: int = 5000
    tri1_path: str = '{exp}/tri1'
    tri2b_path: str = '{exp}/tri2b'
    tri3b_path: str = '{exp}/tri3b'
    score_sh_path: str = '../kaldi/egs/reverb/s5'
    log_models: str = 'tri3b'

class Tasks():
    def __init__(self, logfile, kaldi_path):
        self.logfile = logfile
        self.kaldi_path = kaldi_path

    def run(self, command, check_path=None, desc=None, run_in_kaldi=True, run_in=None):
        if check_path is not None:
            if isinstance(check_path, list):
                run_command = any([not Path(p).exists() for p in check_path])
            else:
                run_command = not Path(check_path).exists()
        else:
            run_command = True
        if run_command:
            if desc is None:
                desc = '[bright_black]' + command + '[/bright_black]'
            with console.status(desc):
                if run_in_kaldi:
                    out = subprocess.run(f'{command}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.kaldi_path)
                elif run_in is None:
                    out = subprocess.run(f'{command}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    out = subprocess.run(f'{command}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=run_in)
            if out.returncode != 0:
                print(f"[red]✕[/red] {desc}")
                print(out.stdout.decode())
                print(out.stderr.decode())
                raise
            else:
                print(f"[green]✓[/green] {desc}")
            return out
        else:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for p in check_path:
                print(f'[blue]{p} already exists[blue]')

run_name = generate_slug(2)

def score_model(task, args, path, name, fmllr=False):
    if name in args.log_models.split(','):
        wandb.init(project="kaldi-synthesised-asr", reinit=True, name=f'{run_name}-{name}')
        wandb.config.update({'model': name})

        graph_path = path / 'graph_tgsmall'
        task.run(
            f"utils/mkgraph.sh {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
            graph_path
        )
        for i, tst in enumerate(args.test_sets):
            graph_test_path = str(graph_path) + f'_{tst}'
            tst_path = args.local_data_path / (tst + '_small')
            if fmllr:
                p_decode = '_fmllr'
            else:
                p_decode = ''
            task.run(
                f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
                graph_test_path
            )
            scoring_path = Path(graph_test_path) / 'scoring_kaldi'
            task.run(
                f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
                scoring_path
            )
            with open(scoring_path / 'best_wer', 'r') as best_wer:
                best_wer = best_wer.read()
            wer = float(re.findall(r'WER (\d+\.\d+)', best_wer)[0])
            ins_err = int(re.findall(r'(\d+) ins', best_wer)[0])
            del_err = int(re.findall(r'(\d+) del', best_wer)[0])
            sub_err = int(re.findall(r'(\d+) sub', best_wer)[0])
            with open(scoring_path / 'wer_details' / 'wer_bootci', 'r') as bootci:
                bootci = bootci.read()
            lower_wer, upper_wer = [round(float(c),2) for c in re.findall(r'Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]', bootci)[0]]
            ci = round((upper_wer - lower_wer) / 2, 2)
            wandb.log({
                f'{tst}/wer': wer,
                f'{tst}/wer_lower': lower_wer,
                f'{tst}/wer_upper': upper_wer,
                f'{tst}/ci_width': ci,
                f'{tst}/ins': ins_err,
                f'{tst}/del': del_err,
                f'{tst}/sub': sub_err,
            })

def run(args):
    os.environ['WANDB_MODE'] = args.wandb_mode

    task = Tasks(args.log_file, args.kaldi_path)
    args.test_sets = args.test_sets.split(',')
    args.lm_names = args.lm_names.split(',')

    for field in fields(args):
        k, v = field.name, getattr(args, field.name)
        if 'path' in field.name:
            if '{data}' in v:
                v = v.replace('{data}', str(args.local_data_path))
            if '{exp}' in v:
                v = v.replace('{exp}', str(args.local_exp_path))
            setattr(args, field.name, Path(v).resolve())

    # download data
    for tst in args.test_sets:
        tst_path = Path(args.data_path) / args.data_name / tst.replace('_', '-')
        task.run(
            f'local/download_and_untar.sh {args.data_path} {args.data_url} {tst}',
            tst_path
        )
            
    # download lm
    task.run(
        f"local/download_lm.sh {args.lm_url} {args.lm_path}",
        args.lm_path
    )

    # prep synth test data
    test_synth_path = Path(args.data_path) / args.train_name / args.synth_test_set
    dest_path = Path(args.local_data_path) / args.synth_test_set
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(test_synth_path, dest_path)
        print(f"[green]✓[/green] loading synthetic test data")
    else:
        print(f'[blue]{dest_path} already exists[blue]')
    args.test_sets = args.test_sets + [args.synth_test_set]

    # prep tts test data
    tts_test_path = args.tts_test_path
    dest_path = Path(args.local_data_path) / 'tts_dev_clean'
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(tts_test_path, dest_path)
        print(f"[green]✓[/green] loading synthetic test data")
    else:
        print(f'[blue]{dest_path} already exists[blue]')
    args.test_sets = args.test_sets + ['tts_dev_clean']

    # prep test data
    for tst in args.test_sets:
        tst_path = Path(args.data_path) / args.data_name / tst.replace('_', '-')
        dest_path = Path(args.local_data_path) / tst
        task.run(
            f"local/data_prep.sh {tst_path} {dest_path}",
            dest_path
        )
        if (Path(dest_path) / 'spk2gender').exists():
            task.run(f"rm {str(Path(dest_path) / 'spk2gender')}", run_in_kaldi=False)
        task.run(
            f'utils/subset_data_dir.sh {dest_path} 100 {str(dest_path) + "_small"}',
            str(dest_path) + "_small"
        )

    # prep train data
    train_path = Path(args.data_path) / args.train_name / args.train_set
    dest_path = Path(args.local_data_path) / args.train_set
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(train_path, dest_path)
        print(f"[green]✓[/green] loading synthetic train data")
    else:
        print(f'[blue]{dest_path} already exists[blue]')

    # create lms
    task.run(
        f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
        args.dict_nosp_path
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
        args.lang_nosp_tmp_path
    )
    task.run(
        f'local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}',
        [
            str(args.lang_nosp_path) + '_test_tgsmall',
            str(args.lang_nosp_path) + '_test_tgmed',
        ]
    )
    if 'tgsmall' not in args.lm_names:
        task.run(f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False)
    if 'tgmed' not in args.lm_names:
        task.run(f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False)
    if 'tglarge' in args.lm_names:
        tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
        task.run(
            f'utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}',
            tglarge_path
        )
    if 'fglarge' in args.lm_names:
        fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
        task.run(
            f'utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}',
            fglarge_path
        )

    # mfccs
    for tst in args.test_sets + [args.train_set]:
        tst_path = args.local_data_path / tst
        exp_path = args.mfcc_path / tst
        task.run(
            f'steps/make_mfcc.sh --cmd {args.train_cmd} --nj {cpus} {tst_path} {exp_path} mfccs',
            exp_path
        )
        task.run(
            f'steps/compute_cmvn_stats.sh {tst_path} {exp_path} mfccs',
            exp_path / f'cmvn_{tst}.log'
        )

    train_path = args.local_data_path / args.train_set

    # mono
    mono_data_path = str(train_path) + '_mono'
    task.run(
        f'utils/subset_data_dir.sh --shortest {train_path} {args.mono_subset} {mono_data_path}',
        mono_data_path
    )
    task.run(
        f'steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}',
        args.mono_path
    )
    score_model(task, args, args.mono_path, 'mono')

    # tri1
    tri1_data_path = str(train_path) + '_tri1'
    task.run(
        f'utils/subset_data_dir.sh {train_path} {args.tri1_subset} {tri1_data_path}',
        tri1_data_path
    )
    tri1_ali_path = str(args.tri1_path) + '_ali'
    task.run(
        f'steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}',
        tri1_ali_path
    )
    task.run(
        f'steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}',
        args.tri1_path
    )
    score_model(task, args, args.tri1_path, 'tri1')

    # tri2b
    tri2b_ali_path = str(args.tri2b_path) + '_ali'
    task.run(
        f'steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}',
        tri2b_ali_path
    )
    task.run(
        f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
        args.tri2b_path
    )
    score_model(task, args, args.tri2b_path, 'tri2b', True)

    # tri3b
    tri3b_ali_path = str(args.tri3b_path) + '_ali'
    task.run(
        f'steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}',
        tri3b_ali_path
    )
    task.run(
        f'steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}',
        args.tri3b_path
    )
    score_model(task, args, args.tri3b_path, 'tri3b', True)
  

  


if __name__ == '__main__':
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    run(args)



