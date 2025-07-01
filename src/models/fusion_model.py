import torch, torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, audio_net, video_net, fusion_dim, num_classes):
        super().__init__()
        self.audio = audio_net
        self.video = video_net
        self.classifier = nn.Sequential(
            nn.Linear(audio_net.out_dim + video_net.out_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_classes)
        )
    def forward(self, wav, wav_lens, vid, *_):
        a = self.audio(wav, wav_lens)
        v = self.video(vid)
        x = torch.cat([a,v], dim=1)
        return self.classifier(x)

class LateFusion(nn.Module):
    def __init__(self, audio_net, video_net, num_classes):
        super().__init__()
        self.audio = audio_net
        self.video = video_net
        self.la = nn.Linear(audio_net.out_dim, num_classes)
        self.lv = nn.Linear(video_net.out_dim, num_classes)
    def forward(self, wav, wav_lens, vid, *_):
        a = self.audio(wav, wav_lens)
        v = self.video(vid)
        return (self.la(a) + self.lv(v)) * 0.5

class HybridFusion(nn.Module):
    def __init__(self, audio_net, video_net, fusion_dim, num_classes):
        super().__init__()
        self.early = EarlyFusion(audio_net, video_net, fusion_dim, num_classes)
        self.late  = LateFusion(audio_net, video_net, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, wav, wav_lens, vid, vid_lens):
        e = self.early(wav, wav_lens, vid, vid_lens)
        l = self.late(wav, wav_lens, vid, vid_lens)
        return self.alpha * e + (1 - self.alpha) * l
