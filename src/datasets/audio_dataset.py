import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance import kaldi

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}


def default_fbank(waveform: torch.Tensor, num_mel_bins: int = 40) -> torch.Tensor:
    return kaldi.fbank(waveform, num_mel_bins=num_mel_bins)


class RAVDESSSpeechDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        emotions=None,
        intensity_levels=None,
        transform=None,
        actors=None
    ):
        self.emotion_list = list(emotions) if emotions else list(EMOTION_MAP.values())
        self.emotion2idx = {emo: i for i, emo in enumerate(self.emotion_list)}
        self.emotions = set(self.emotion_list)
        self.intensities = set(intensity_levels) if intensity_levels else set(INTENSITY_MAP.values())
        self.transform = transform if transform is not None else default_fbank

        if actors:
            self.actors = set(actors)
        else:
            discovered = []
            for name in os.listdir(base_dir):
                if name.startswith("Actor_"):
                    _, aid = name.split("_")
                    discovered.append(aid)
            if not discovered:
                raise RuntimeError(f"No Actor_*/ under {base_dir}")
            self.actors = set(discovered)

        self.files = []
        self.labels = []
        for root, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if not fname.lower().endswith(".wav"):
                    continue
                parts = fname[:-4].split("-")
                if len(parts) != 7:
                    continue
                modality, _, emo_c, int_c, *_ , actor = parts
                if modality != "03" or actor not in self.actors:
                    continue
                emo = EMOTION_MAP.get(emo_c)
                intensity = INTENSITY_MAP.get(int_c)
                if emo not in self.emotions or intensity not in self.intensities:
                    continue

                self.files.append(os.path.join(root, fname))
                self.labels.append(self.emotion2idx[emo])

        if not self.files:
            raise RuntimeError(f"No audio files in {base_dir} for actors {sorted(self.actors)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            import soundfile as sf
            import numpy as np
            data_np, sr = sf.read(path)
            if data_np.ndim == 1:
                data_np = np.expand_dims(data_np, 0)
            else:
                data_np = data_np.T
            waveform = torch.from_numpy(data_np)

        feat = self.transform(waveform, num_mel_bins=40)
        return feat, label
