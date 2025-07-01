import torch, torch.nn as nn

def pad_collate_audio(batch):
    feats, lens, *_ = batch
    lens = torch.tensor(lens)
    padded = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return padded, lens

def collate_multimodal(batch):
    feats, lens, vids, labels = zip(*batch)

    lens = torch.tensor(lens)
    feats = nn.utils.rnn.pad_sequence(feats, batch_first=True)

    vids = torch.stack(vids)
    labels = torch.tensor(labels)
    return feats, lens, vids, labels
