from pathlib import Path
from dataclasses import dataclass, fields
import subprocess
from typing import List
import os
from multiprocessing import cpu_count
import re
import shutil
import sys

from transformers import HfArgumentParser
from rich import print as rprint
from rich.console import Console
from rich.progress import track
import wandb
from coolname import generate_slug

from prepare_synth import load_synth

console = Console()
cpus = cpu_count()


@dataclass
class Args:
    wandb_mode: str = "offline"
    kaldi_path: str = "../kaldi/egs/librispeech/s5"
    data_path: str = "../Data/"
    data_url: str = "www.openslr.org/resources/12"
    data_name: str = "LibriSpeech"
    lm_path: str = "{data}/local/lm"
    dict_path: str = "{data}/local/dict"
    dict_nosp_path: str = "{data}/local/dict_nosp"
    lang_path: str = "{data}/lang"
    lang_tmp_path: str = "{data}/local/lang_tmp"
    lang_nosp_path: str = "{data}/lang_nosp"
    lang_nosp_tmp_path: str = "{data}/local/lang_tmp_nosp"
    lm_url: str = "www.openslr.org/resources/11"
    log_file: str = "train.log"
    test_sets: str = "dev_clean,test_clean"
    tts_test_path: str = "../alignments/test"
    local_data_path: str = "data"
    local_exp_path: str = "exp"
    mfcc_path: str = "{data}/mfcc"
    train_name: str = "Synthesised"
    train_set: str = "n_topline"
    synth_test_set: str = "dev_synth"
    train_cmd: str = "run.pl"
    lm_names: str = "tgsmall,tgmed"  # tglarge,fglarge
    mfcc_path: str = "{exp}/make_mfcc"
    mono_subset: int = 2000
    mono_path: str = "{exp}/mono"
    tri1_subset: int = 5000
    tri1_path: str = "{exp}/tri1"
    tri2b_path: str = "{exp}/tri2b"
    tri3b_path: str = "{exp}/tri3b"
    log_models: str = "all"
    nnet_path: str = "../kaldi/egs/librispeech/s5"
    nnet_script_path: str = (
        "../kaldi/egs/librispeech/s5/local/chain/tuning/run_tdnn_1d.sh"
    )
    tdnn_path: str = "{exp}/tdnn"
    verbose: bool = False
    run_name: str = None
    clean_stages: str = "none"


class Tasks:
    def __init__(self, logfile, kaldi_path):
        self.logfile = logfile
        self.kaldi_path = kaldi_path

    def execute(self, command, **kwargs):
        popen = subprocess.Popen(
            f"{command}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        for line in popen.stdout:
            yield line.decode()
        for line in popen.stderr:
            yield line.decode()
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def run(
        self,
        command,
        check_path=None,
        desc=None,
        run_in_kaldi=True,
        run_in=None,
        clean=False,
    ):
        if check_path is not None:
            if isinstance(check_path, list):
                run_command = any([not Path(p).exists() for p in check_path])
            else:
                run_command = not Path(check_path).exists()
        else:
            run_command = True
        if clean:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for c_p in check_path:
                c_p = Path(c_p)
                if c_p.is_dir():
                    shutil.rmtree(c_p)
                if c_p.is_file():
                    os.remove(c_p)
            run_command = True
        if run_command:
            if desc is None:
                desc = "[bright_black]" + command + "[/bright_black]"
            with console.status(desc):
                if run_in_kaldi:
                    for path in self.execute(command, cwd=self.kaldi_path):
                        if args.verbose:
                            print(path, end="")
                elif run_in is None:
                    for path in self.execute(command):
                        if args.verbose:
                            print(path, end="")
                else:
                    for path in self.execute(command, cwd=run_in):
                        if args.verbose:
                            print(path, end="")
            rprint(f"[green]✓[/green] {desc}")
        else:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for p in check_path:
                rprint(f"[blue]{p} already exists[blue]")


run_name = generate_slug(2)


def score_model(task, args, path, name, fmllr=False, lang_nosp=True, nnet=False):
    if args.run_name is not None:
        run_name = args.run_name
    if args.log_models == "all" or name in args.log_models.split(","):
        wandb.init(
            project="kaldi-synthesised-asr",
            reinit=True,
            name=f"{run_name}-{name}",
            group=run_name,
        )
        wandb.config.update(
            {
                "model": name,
                "group_name": run_name,
            }
        )
        if nnet:
            mkgraph_args = "--self-loop-scale 1.0 --remove-oov"
        else:
            mkgraph_args = ""
        if lang_nosp:
            graph_path = path / "graph_tgsmall"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        else:
            graph_path = path / "graph_tgsmall_sp"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        for i, tst in enumerate(args.test_sets):
            graph_test_path = str(graph_path) + f"_{tst}"
            tst_path = args.local_data_path / (tst + "_small")
            if fmllr:
                p_decode = "_fmllr"
            else:
                p_decode = ""
            if not nnet:
                task.run(
                    f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
                    graph_test_path,
                )
            else:
                graph_test_path = path / f"decode_{tst}_small_tgsmall"
                task.run(
                    f"steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} --online-ivector-dir exp/nnet3_cleaned/ivectors_{tst}_small_hires {graph_path} {tst_path}_hires {graph_test_path}",
                    graph_test_path,
                )
            scoring_path = Path(graph_test_path) / "scoring_kaldi"
            task.run(
                f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
                scoring_path,
            )
            with open(scoring_path / "best_wer", "r") as best_wer:
                best_wer = best_wer.read()
            wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
            ins_err = int(re.findall(r"(\d+) ins", best_wer)[0])
            del_err = int(re.findall(r"(\d+) del", best_wer)[0])
            sub_err = int(re.findall(r"(\d+) sub", best_wer)[0])
            with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
                bootci = bootci.read()
            lower_wer, upper_wer = [
                round(float(c), 2)
                for c in re.findall(
                    r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci
                )[0]
            ]
            ci = round((upper_wer - lower_wer) / 2, 2)
            wandb.log(
                {
                    f"{tst}/wer": wer,
                    f"{tst}/wer_lower": lower_wer,
                    f"{tst}/wer_upper": upper_wer,
                    f"{tst}/ci_width": ci,
                    f"{tst}/ins": ins_err,
                    f"{tst}/del": del_err,
                    f"{tst}/sub": sub_err,
                }
            )


def run(args):
    os.environ["WANDB_MODE"] = args.wandb_mode

    task = Tasks(args.log_file, args.kaldi_path)
    args.test_sets = args.test_sets.split(",")
    args.lm_names = args.lm_names.split(",")
    if "," in args.clean_stages:
        args.clean_stages = args.clean_stages.split(",")
    else:
        args.clean_stages = [args.clean_stages]

    for field in fields(args):
        k, v = field.name, getattr(args, field.name)
        if "path" in field.name:
            if "{data}" in v:
                v = v.replace("{data}", str(args.local_data_path))
            if "{exp}" in v:
                v = v.replace("{exp}", str(args.local_exp_path))
            setattr(args, field.name, Path(v).resolve())

    # download data
    for tst in args.test_sets:
        tst_path = Path(args.data_path) / args.data_name / tst.replace("_", "-")
        task.run(
            f"local/download_and_untar.sh {args.data_path} {args.data_url} {tst}",
            tst_path,
        )

    # download lm
    task.run(f"local/download_lm.sh {args.lm_url} {args.lm_path}", args.lm_path)

    # prep synth test data
    test_synth_path = Path(args.data_path) / args.train_name / args.synth_test_set
    dest_path = Path(args.local_data_path) / args.synth_test_set
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(test_synth_path, dest_path)
        rprint(f"[green]✓[/green] loading synthetic test data")
    else:
        rprint(f"[blue]{dest_path} already exists[blue]")
    args.test_sets = args.test_sets + [args.synth_test_set]

    # prep tts test data
    tts_test_path = args.tts_test_path
    dest_path = Path(args.local_data_path) / "tts_dev_clean"
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(tts_test_path, dest_path)
        rprint(f"[green]✓[/green] loading synthetic test data")
    else:
        rprint(f"[blue]{dest_path} already exists[blue]")
    args.test_sets = args.test_sets + ["tts_dev_clean"]

    # prep train data
    train_path = Path(args.data_path) / args.train_name / args.train_set
    dest_path = Path(args.local_data_path) / args.train_set
    if not dest_path.exists():
        dest_path.mkdir()
        load_synth(train_path, dest_path)
        rprint(f"[green]✓[/green] loading synthetic train data")
    else:
        rprint(f"[blue]{dest_path} already exists[blue]")

    # prep test data
    for tst in args.test_sets:
        tst_path = Path(args.data_path) / args.data_name / tst.replace("_", "-")
        dest_path = Path(args.local_data_path) / tst
        task.run(f"local/data_prep.sh {tst_path} {dest_path}", dest_path)
        if (Path(dest_path) / "spk2gender").exists():
            task.run(f"rm {str(Path(dest_path) / 'spk2gender')}", run_in_kaldi=False)

    # create lms
    task.run(
        f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
        args.dict_nosp_path,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
        args.lang_nosp_tmp_path,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}",
        [
            str(args.lang_nosp_path) + "_test_tgsmall",
            str(args.lang_nosp_path) + "_test_tgmed",
        ],
    )
    if "tgsmall" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False
        )
    if "tgmed" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False
        )
    if "tglarge" in args.lm_names:
        tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}",
            tglarge_path,
        )
    if "fglarge" in args.lm_names:
        fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}",
            fglarge_path,
        )

    # mfccs
    for tst in args.test_sets + [args.train_set]:
        tst_path = args.local_data_path / tst
        exp_path = args.mfcc_path / tst
        task.run(
            f"steps/make_mfcc.sh --cmd {args.train_cmd} --nj {cpus} {tst_path} {exp_path} mfccs",
            tst_path / "feats.scp",
        )
        task.run(
            f"steps/compute_cmvn_stats.sh {tst_path} {exp_path} mfccs",
            exp_path / f"cmvn_{tst}.log",
        )

    # generate subsets
    for tst in args.test_sets:
        dest_path = Path(args.local_data_path) / tst
        task.run(
            f'utils/subset_data_dir.sh {dest_path} 100 {str(dest_path) + "_small"}',
            str(dest_path) + "_small",
        )

    train_path = args.local_data_path / args.train_set

    # mono
    mono_data_path = str(train_path) + "_mono"
    clean_mono = "mono" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"utils/subset_data_dir.sh --shortest {train_path} {args.mono_subset} {mono_data_path}",
        mono_data_path,
    )
    task.run(
        f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}",
        args.mono_path,
        clean=clean_mono,
    )
    score_model(task, args, args.mono_path, "mono")

    # tri1
    tri1_data_path = str(train_path) + "_tri1"
    clean_tri1 = "tri1" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"utils/subset_data_dir.sh {train_path} {args.tri1_subset} {tri1_data_path}",
        tri1_data_path,
    )
    tri1_ali_path = str(args.tri1_path) + "_ali"
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}",
        tri1_ali_path,
        clean=clean_tri1,
    )
    task.run(
        f"steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}",
        args.tri1_path,
        clean=clean_tri1,
    )
    score_model(task, args, args.tri1_path, "tri1", True)

    # tri2b
    tri2b_ali_path = str(args.tri2b_path) + "_ali"
    clean_tri2b = "tri2b" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}",
        tri2b_ali_path,
        clean_tri2b,
    )

    task.run(
        f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
        args.tri2b_path,
    )
    score_model(task, args, args.tri2b_path, "tri2b", True)

    # tri3b
    tri3b_ali_path = str(args.tri3b_path) + "_ali"
    task.run(
        f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}",
        tri3b_ali_path,
    )
    task.run(
        f"steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}",
        args.tri3b_path,
    )
    score_model(task, args, args.tri3b_path, "tri3b", True)

    # recompute lm
    task.run(
        f"steps/get_prons.sh --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri3b_path}",
        [
            args.tri3b_path / "pron_counts_nowb.txt",
            args.tri3b_path / "sil_counts_nowb.txt",
            args.tri3b_path / "pron_bigram_counts_nowb.txt",
        ],
    )
    task.run(
        f'utils/dict_dir_add_pronprobs.sh --max-normalize true {args.dict_nosp_path} {args.tri3b_path / "pron_counts_nowb.txt"} {args.tri3b_path / "sil_counts_nowb.txt"} {args.tri3b_path / "pron_bigram_counts_nowb.txt"} {args.dict_path}',
        args.dict_path,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_path} "<UNK>" {args.lang_tmp_path} {args.lang_path}',
        args.lang_path,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_path} {args.lm_path}",
        [
            str(args.lang_path) + "_test_tgsmall",
            str(args.lang_path) + "_test_tgmed",
        ],
    )
    score_model(task, args, args.tri3b_path, "tri3b-probs", True, False)

    # train tdnn
    tri3b_ali_path = str(args.tri3b_path) + f"_ali_{train_path.name}_sp"
    task.run(
        f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_path} {args.tri3b_path} {tri3b_ali_path}",
        tri3b_ali_path,
    )
    if not Path(args.nnet_path / "data").exists():
        os.symlink(str(args.local_data_path), str(args.nnet_path / "data"))
    if not Path(args.nnet_path / "exp").exists():
        os.symlink(str(args.local_exp_path), str(args.nnet_path / "exp"))
    task.run(
        f"sed -i 's/queue.pl/run.pl/' {args.nnet_path / 'cmd.sh'}", run_in_kaldi=False
    )
    task.run(
        f"sed -i 's/-le 24/-le -1/' {args.nnet_script_path}", run_in_kaldi=False
    )  # move lower
    test_set_string = " ".join([f"{tst}_small" for tst in args.test_sets])
    ivec_path = args.nnet_path / "local" / "nnet3" / "run_ivector_common.sh"
    task.run(
        f"sed -i 's/60000/1000/' {ivec_path}", run_in_kaldi=False
    )  # replace with \d+ and re
    ivec_txt = "".join(open(ivec_path, "r").readlines())
    ivec_txt = re.sub(
        r"(for data.* in \${train_set}_sp )([^;]+)(;)",
        r"\1" + test_set_string + r"\3",
        ivec_txt,
    )
    ivec_txt = re.sub(
        r"(for data in )([^;]+)(;)", r"\1" + test_set_string + r"\3", ivec_txt
    )
    with open(ivec_path, "w") as ivec_file:
        ivec_file.write(ivec_txt)

    # set nnet config
    nnet_txt = "".join(open(args.nnet_script_path, "r").readlines())
    network_re = re.compile(
        r"(<<EOF > \$dir/configs/network\.xconfig((.|\n)*)[^<]EOF)", re.MULTILINE
    )
    replacement = "<<EOF > $dir/configs/network.xconfig\n"
    for line in open("network.txt", "r"):
        replacement += f"  {line}"
    replacement += "\nEOF"
    nnet_txt = network_re.sub(replacement, nnet_txt)
    nnet_txt = re.sub(r"num-jobs-final \d+", f"num-jobs-final 8", nnet_txt)

    # disable mkgraph and decoding stage
    for stage in ["-le 16", "-le 17", "-le 18"]:
        nnet_txt = nnet_txt.replace(stage, "-le -1")

    with open(args.nnet_script_path, "w") as nnet_file:
        nnet_file.write(nnet_txt)

    train_sp = args.local_data_path / (train_path.name + "_sp")
    train_ali = args.local_exp_path / f"{args.tri3b_path.name}_ali_{train_path.name}_sp"

    task.run(
        f"{args.nnet_script_path} --train_set {train_path.name} --gmm {args.tri3b_path.name}",
        [
            train_sp,
            train_ali,
            args.local_exp_path / "nnet3_cleaned",
            args.local_exp_path / "chain_cleaned",
        ],
        clean=False,
    )
    score_model(
        task,
        args,
        args.local_exp_path / "chain_cleaned" / "tdnn_1d_sp",
        "tdnn",
        True,
        False,
        nnet=True,
    )


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    run(args)
