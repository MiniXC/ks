from pathlib import Path
from shutil import copy
from tqdm.auto import tqdm

target = Path("../TTS/Data") / "real"
target.mkdir(exist_ok=True)

src_paths = [Path("../TTS/data/train-clean-100-aligned"), Path("../TTS/data/train-clean-360-aligned")]

for wav in tqdm(list(Path("../TTS/generated/base/").rglob("*.wav"))):
    if "original" in str(wav):
        continue
    speaker_dir = target / wav.parent.name
    speaker_dir.mkdir(exist_ok=True)
    lab = str(wav).replace(".wav", ".lab")
    copy(lab, target / wav.parent.name / Path(lab).name)
    for src_path in src_paths:
        if (src_path / wav.parent.name / wav.name).exists():
            copy(src_path / wav.parent.name / wav.name, target / wav.parent.name / wav.name)
            break