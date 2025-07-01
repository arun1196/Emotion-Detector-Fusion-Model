import torch
import torch.nn as nn

class VoiceLSTM(nn.Module):
    """
    Bi-LSTM over fbank features → fixed-size embedding.
    """
    def __init__(self, in_dim=40, hidden_dim=64, num_layers=1, out_features=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, out_features)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h)


def build_audio_net(out_dim=128, in_dim=40, hidden_dim=64, num_layers=1):
    """
    Factory to create the audio embedding network.
    Attaches an `out_dim` attribute for fusion sizing.
    """
    net = VoiceLSTM(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_features=out_dim
    )
    net.out_dim = out_dim
    return net
