import os
import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from src.datasets.audio_dataset import RAVDESSSpeechDataset
from src.datasets.video_dataset import RAVDESSVideoDataset
from src.models.audio_model      import build_audio_net
from src.models.video_model      import build_video_net
from src.models.fusion_model     import EarlyFusion, LateFusion, HybridFusion

def plot_probs(probs: np.ndarray, labels: list, title: str = "Emotion probabilities"):
    plt.figure(figsize=(8,4))
    x = np.arange(len(labels))
    plt.bar(x, probs)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()

def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

def plot_tsne(proj: np.ndarray, label_indices: list, labels: list, title: str = "t-SNE of Fusion Embeddings"):
    plt.figure(figsize=(8,6))
    for idx, emo in enumerate(labels):
        mask = [li == idx for li in label_indices]
        plt.scatter(proj[mask,0], proj[mask,1], label=emo, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(title)
    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and optionally evaluation with a trained fusion model")
    parser.add_argument("--audio",    required=True, help="Path to the .wav file")
    parser.add_argument("--video",    required=True, help="Path to the .mp4 file")
    parser.add_argument("--model",    required=True, help="Path to fusion_model.pth")
    parser.add_argument("--fusion",   required=True, choices=["early","late","hybrid"], help="Fusion type")
    parser.add_argument("--outdim",   type=int, default=128, help="Embedding size (must match training)")
    parser.add_argument("--num_cls",  type=int, default=7,   help="Number of emotion classes")
    parser.add_argument("--config",   default="configs/fusion_config.yaml", help="Path to your training config YAML")
    parser.add_argument("--evaluate", action="store_true",
                        help="If set, run evaluation over the entire dataset (confusion matrix + t-SNE)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    emotion_list = cfg["audio"]["emotions"]

    audio_ds = RAVDESSSpeechDataset(
        base_dir=cfg["audio"]["base_dir"],
        emotions=emotion_list,
        intensity_levels=cfg["audio"]["intensity_levels"],
        transform=None,
        actors=cfg["audio"].get("actors", None)
    )
    video_ds = RAVDESSVideoDataset(
        base_dir=cfg["video"]["base_dir"],
        emotions=cfg["video"]["emotions"],
        intensity_levels=cfg["video"]["intensity_levels"],
        num_frames=cfg["video"]["num_frames"],
        actors=cfg["video"].get("actors", None)
    )

    audio_net = build_audio_net(out_dim=args.outdim)
    video_net = build_video_net(out_dim=args.outdim)
    FusionCls = {"early": EarlyFusion,
                 "late":  LateFusion,
                 "hybrid": HybridFusion}[args.fusion]
    model = FusionCls(audio_net, video_net,
                      fusion_dim=args.outdim,
                      num_classes=args.num_cls)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    def stem(path): return os.path.basename(path).rsplit(".",1)[0]

    key_a = stem(args.audio)
    key_v = stem(args.video)
    try:
        idx_a = next(i for i, p in enumerate(audio_ds.files) if stem(p) == key_a)
        idx_v = next(i for i, p in enumerate(video_ds.files) if stem(p) == key_v)
    except StopIteration:
        raise FileNotFoundError("Could not find the given audio/video in the configured dataset.")

    feat, true_idx_a = audio_ds[idx_a]
    vid_tensor, _     = video_ds[idx_v]
    wav  = feat.unsqueeze(0)
    wlen = torch.tensor([wav.shape[1]])
    vbat = vid_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(wav, wlen, vbat, None)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred   = int(probs.argmax())

    print(f"Predicted emotion/spoof: {emotion_list[pred]}")

    # Plot
    plot_probs(probs, emotion_list, title=f"Predicted: {emotion_list[pred]}")
    plt.savefig("emotion_prob.png")
    plt.close()
    print("Saved per-sample probabilities to emotion_prob.png")

    if args.evaluate:
        y_true, y_pred = [], []
        for i in range(len(audio_ds)):
            feat_i, true_idx = audio_ds[i]
            vid_i, _         = video_ds[i]
            wav_i  = feat_i.unsqueeze(0)
            wlen_i = torch.tensor([wav_i.shape[1]])
            vbat_i = vid_i.unsqueeze(0)
            with torch.no_grad():
                out_logits = model(wav_i, wlen_i, vbat_i, None)
                y_pred.append(int(out_logits.argmax(dim=1).item()))
                y_true.append(true_idx)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_list))))
        plot_confusion_matrix(cm, emotion_list)
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("Saved confusion matrix to confusion_matrix.png")

        # 2) t-SNE of fusion embeddings
        feats, labels_tsne = [], []
        for i in range(len(audio_ds)):
            feat_i, true_idx = audio_ds[i]
            vid_i, _         = video_ds[i]
            wav_i  = feat_i.unsqueeze(0)
            wlen_i = torch.tensor([wav_i.shape[1]])
            vbat_i = vid_i.unsqueeze(0)
            with torch.no_grad():
                emb, _ = model.encode(wav_i, wlen_i, vbat_i, None)
            feats.append(emb.squeeze(0).cpu().numpy())
            labels_tsne.append(true_idx)

        feats = np.vstack(feats)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        proj = tsne.fit_transform(feats)

        plot_tsne(proj, labels_tsne, emotion_list)
        plt.savefig("tsne_embeddings.png")
        plt.close()
        print("Saved t-SNE plot to tsne_embeddings.png")
