import os
import torch
from torch.utils.data import Dataset
from .audio_dataset import RAVDESSSpeechDataset
from .video_dataset import RAVDESSVideoDataset

class RAVDESSMultiModalDataset(Dataset):
    """
    Pairs up audio-only (.wav, modality 03) and audio-video (.mp4, modality 01)
    by matching the utterance identifier (parts[2:] of filename).
    Returns (feat, length, video_tensor, label).
    """
    def __init__(self, audio_cfg, video_cfg):
        self.audio_ds = RAVDESSSpeechDataset(**audio_cfg)
        self.video_ds = RAVDESSVideoDataset(**video_cfg)
        self.audio_map = {}
        for i, path in enumerate(self.audio_ds.files):
            fname = os.path.basename(path)[:-4]
            parts = fname.split("-")
            if len(parts) != 7:
                continue
            key = "-".join(parts[2:])
            self.audio_map[key] = i

        pairs = []
        for vid_idx, vpath in enumerate(self.video_ds.files):
            fname = os.path.basename(vpath)[:-4]
            parts = fname.split("-")
            if len(parts) != 7:
                continue
            key = "-".join(parts[2:])
            if key in self.audio_map:
                pairs.append((self.audio_map[key], vid_idx))

        if not pairs:
            raise RuntimeError("No matching audio/video files found.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_idx, v_idx = self.pairs[idx]
        feat, lbl_a = self.audio_ds[a_idx]
        video, lbl_v = self.video_ds[v_idx]
        assert lbl_a == lbl_v, "Audio/video label mismatch!"
        return feat, feat.size(0), video, lbl_a
