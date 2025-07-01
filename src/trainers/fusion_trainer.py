import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
from src.utils.collate_fns import collate_multimodal

class FusionTrainer:
    def __init__(self, model, cfg, output_dir):
        self.device     = torch.device(cfg.device)
        self.model      = model.to(self.device)
        self.crit       = torch.nn.CrossEntropyLoss()
        self.opt        = optim.Adam(model.parameters(), lr=cfg.lr)
        self.batch      = cfg.batch_size
        self.epochs     = cfg.epochs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fit(self, dataset):
        # Split 80/20
        n_train = int(0.8 * len(dataset))
        train_ds, val_ds = random_split(dataset, [n_train, len(dataset)-n_train])
        train_loader = DataLoader(train_ds, batch_size=self.batch, shuffle=True,
                                  collate_fn=collate_multimodal, num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch, shuffle=False,
                                  collate_fn=collate_multimodal, num_workers=4)

        train_losses, val_accs = [], []

        # Training loop
        for e in range(1, self.epochs+1):
            self.model.train()
            running_loss = 0.0
            for wav, wlen, vid, lbl in tqdm(train_loader, desc=f"Epoch {e} [train]"):
                wav, wlen, vid, lbl = [t.to(self.device) for t in (wav,wlen,vid,lbl)]
                self.opt.zero_grad()
                logits = self.model(wav, wlen, vid, None)
                loss   = self.crit(logits, lbl)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {e} | Train loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            correct = total = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for wav, wlen, vid, lbl in val_loader:
                    wav, wlen, vid, lbl = [t.to(self.device) for t in (wav,wlen,vid,lbl)]
                    logits = self.model(wav, wlen, vid, None)
                    preds  = logits.argmax(dim=1)
                    correct += (preds == lbl).sum().item()
                    total   += lbl.size(0)
                    all_preds .extend(preds.cpu().numpy())
                    all_labels.extend(lbl.cpu().numpy())
            acc = 100 * correct / total
            val_accs.append(acc)
            print(f"Epoch {e} | Val acc   : {acc:.2f}%")

        epochs = np.arange(1, self.epochs+1)

        # 1. Training Loss Curve
        plt.figure()
        plt.plot(epochs, train_losses, marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'train_loss.png'))
        plt.close()

        # 2. Validation Accuracy Curve
        plt.figure()
        plt.plot(epochs, val_accs, marker='o')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'val_accuracy.png'))
        plt.close()

        # 3. Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

        # 4. Classification Report
        report_txt = classification_report(all_labels, all_preds)
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report_txt)

        # 5. Precision / Recall / F1 / Support per class
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )
        with open(os.path.join(self.output_dir, 'metrics_summary.txt'), 'w') as f:
            f.write("Class  Precision  Recall  F1-Score  Support\n")
            for i, (p, r, f1s, s) in enumerate(zip(precision, recall, f1, support)):
                f.write(f"{i:>5}  {p:9.2f}  {r:6.2f}  {f1s:8.2f}  {s:7}\n")

        print(f"\nAll plots and metrics saved to `{self.output_dir}`")
