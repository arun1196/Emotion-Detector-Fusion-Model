import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class VideoEmotionNet(nn.Module):
    """
    Takes a batch of videos [B, T, C, H, W], runs each frame
    through a pretrained facenet, then a bi-LSTM, and outputs
    a fixed-size embedding of size `out_dim`.
    """
    def __init__(self, feat_dim=512, lstm_hidden=64, num_layers=1, out_dim=128):
        super().__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            feat_dim,
            lstm_hidden,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        feats = self.backbone(frames)
        feats = feats.view(B, T, -1)
        lengths = torch.full((B,), T, dtype=torch.long, device=feats.device)
        packed = nn.utils.rnn.pack_padded_sequence(
            feats,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        h_f = h_n[-2]
        h_b = h_n[-1]
        h   = torch.cat([h_f, h_b], dim=1)
        return self.fc(h)

def build_video_net(out_dim=128, feat_dim=512, lstm_hidden=64, num_layers=1):
    """
    Factory for the video branch.  Usage in train.py:
        video_net = build_video_net(out_dim=128)
    """
    net = VideoEmotionNet(
        feat_dim=feat_dim,
        lstm_hidden=lstm_hidden,
        num_layers=num_layers,
        out_dim=out_dim
    )
    return net
