import os
import torch
from torch.utils.data import Dataset
import cv2
from facenet_pytorch import MTCNN
import torchvision.transforms as T
from .audio_dataset import EMOTION_MAP, INTENSITY_MAP

class RAVDESSVideoDataset(Dataset):
    def __init__(
        self,
        base_dir,
        emotions=None,
        intensity_levels=None,
        num_frames=8,
        mtcnn_kwargs=None,
        actors=None
    ):
        self.emotion_list = list(emotions) if emotions else list(EMOTION_MAP.values())
        self.emotion2idx = {emo: i for i, emo in enumerate(self.emotion_list)}
        self.emotions = set(self.emotion_list)

        self.intensities = set(intensity_levels) if intensity_levels else set(INTENSITY_MAP.values())
        self.num_frames = num_frames

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

        self.mtcnn = MTCNN(**(mtcnn_kwargs or {}), keep_all=False)
        self.post_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor(),
        ])

        self.files = []
        self.labels = []
        for root, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if not fname.lower().endswith(".mp4"):
                    continue
                parts = fname[:-4].split("-")
                if len(parts) != 7:
                    continue
                modality, _, emo_c, int_c, *_ , actor = parts
                if modality != "01" or actor not in self.actors:
                    continue
                emo = EMOTION_MAP.get(emo_c)
                intensity = INTENSITY_MAP.get(int_c)
                if emo not in self.emotions or intensity not in self.intensities:
                    continue

                self.files.append(os.path.join(root, fname))
                self.labels.append(self.emotion2idx[emo])

        if not self.files:
            raise RuntimeError(f"No video files in {base_dir} for actors {sorted(self.actors)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.labels[idx]
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = torch.linspace(0, total-1, steps=self.num_frames).long().tolist()

        frames = []
        for fno in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = self.mtcnn(img)
            if face is None:
                h, w, _ = img.shape
                c = min(h,w)
                y0, x0 = (h-c)//2, (w-c)//2
                crop = img[y0:y0+c, x0:x0+c]
                face = self.post_transform(crop)
            else:
                face = self.post_transform(face)
            frames.append(face)

        cap.release()
        if len(frames) < self.num_frames:
            pad = [torch.zeros_like(frames[0])] * (self.num_frames - len(frames))
            frames += pad

        return torch.stack(frames), label
