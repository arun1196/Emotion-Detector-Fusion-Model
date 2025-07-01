import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from src.datasets.multimodal_dataset import RAVDESSMultiModalDataset
from src.models.audio_model      import build_audio_net
from src.models.video_model      import build_video_net
from src.models.fusion_model     import EarlyFusion, LateFusion, HybridFusion
from src.trainers.fusion_trainer import FusionTrainer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir",   required=True,
                   help="path to Audio_Speech_Actors_01-04")
    p.add_argument("--video-dir",   required=True,
                   help="path to Video_Speech_Actors_01-04")
    p.add_argument("--fusion",      choices=["early","late","hybrid"],
                   required=True)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--output-dir",  required=True)
    args = p.parse_args()

    audio_cfg = {
        "base_dir":           args.audio_dir,
        "emotions":           ["calm","happy","sad","angry","fearful","surprised","disgust"],
        "intensity_levels":   ["normal","strong"],
        "transform":          None,
        "actors":             ["01","02","03","04"],
    }
    video_cfg = {
        "base_dir":           args.video_dir,
        "emotions":           ["calm","happy","sad","angry","fearful","surprised","disgust"],
        "intensity_levels":   ["normal","strong"],
        "num_frames":         8,
        "mtcnn_kwargs":       None,
        "actors":             ["01","02","03","04"],
    }

    dataset = RAVDESSMultiModalDataset(audio_cfg, video_cfg)

    audio_net = build_audio_net(out_dim=128)
    video_net = build_video_net(out_dim=128)

    FusionCls = {"early":   EarlyFusion,
                 "late":    LateFusion,
                 "hybrid":  HybridFusion}[args.fusion]

    model = FusionCls(
        audio_net,
        video_net,
        fusion_dim=128,
        num_classes= len(audio_cfg["emotions"])
    )

    trainer = FusionTrainer(
        model,
        cfg=argparse.Namespace(
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs
        ),
        
        output_dir=args.output_dir
    )
    trainer.fit(dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "fusion_model.pth"))
    print(f"Model saved to {args.output_dir}/fusion_model.pth")
