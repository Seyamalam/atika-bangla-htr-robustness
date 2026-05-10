from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "bn_htrd_lines"
RESULTS = ROOT / "results"


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def choose_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    # CTCLoss is not implemented on MPS in the current PyTorch build, and the
    # CPU fallback can be slower and harder to reason about. Keep this baseline
    # on CPU by default for reproducible CTC training.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LineDataset(Dataset):
    def __init__(self, csv_path: Path, vocab: dict[str, int], image_width: int, max_rows: int | None = None):
        df = pd.read_csv(csv_path)
        if max_rows:
            df = df.head(max_rows)
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.image_width = image_width

    def __len__(self) -> int:
        return len(self.df)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.vocab[ch] for ch in text if ch in self.vocab], dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = ROOT / row["image_path"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        target_h = 48
        h, w = image.shape
        scale = target_h / max(h, 1)
        new_w = max(1, min(self.image_width, int(w * scale)))
        image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, self.image_width), 255, dtype=np.uint8)
        canvas[:, :new_w] = image
        canvas = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(canvas).unsqueeze(0)
        text = str(row["text"])
        return tensor, self.encode(text), text, row["line_id"]


def collate(batch):
    images, targets, texts, line_ids = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
    return images, targets, target_lengths, list(texts), list(line_ids)


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feats = self.cnn(x)
        feats = feats.mean(dim=2).permute(0, 2, 1)
        out, _ = self.rnn(feats)
        logits = self.fc(out)
        return logits.permute(1, 0, 2)


def decode_batch(logits: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    pred = logits.argmax(dim=2).permute(1, 0).detach().cpu().numpy()
    decoded = []
    for seq in pred:
        chars = []
        prev = 0
        for idx in seq:
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        decoded.append("".join(chars))
    return decoded


def evaluate(model, loader, criterion, device, idx_to_char, max_batches: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    examples = []
    batches = 0
    with torch.no_grad():
        for images, targets, target_lengths, texts, line_ids in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths.to(device))
            total_loss += float(loss.detach().cpu())
            preds = decode_batch(logits, idx_to_char)
            for line_id, truth, pred in zip(line_ids, texts, preds):
                total_char_dist += levenshtein(pred, truth)
                total_chars += max(1, len(truth))
                total_word_dist += levenshtein(pred.split(), truth.split())
                total_words += max(1, len(truth.split()))
                if len(examples) < 20:
                    examples.append({"line_id": line_id, "truth": truth, "pred": pred})
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return {
        "loss": total_loss / max(1, batches),
        "cer": total_char_dist / max(1, total_chars),
        "wer": total_word_dist / max(1, total_words),
        "examples": examples,
        "batches": batches,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    random.seed(42)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    idx_to_char = {idx: ch for ch, idx in vocab.items() if idx != 0}
    device = choose_device(args.device)
    train_ds = LineDataset(DATA / "train.csv", vocab, args.image_width, args.max_train_rows)
    val_ds = LineDataset(DATA / "val.csv", vocab, args.image_width, args.max_val_rows)
    test_ds = LineDataset(DATA / "test.csv", vocab, args.image_width, args.max_test_rows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = CRNN(len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for images, targets, target_lengths, _, _ in tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}"):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
            if args.max_train_batches and batches >= args.max_train_batches:
                break
        val = evaluate(model, val_loader, criterion, device, idx_to_char, max_batches=args.max_eval_batches)
        row = {"epoch": epoch, "train_loss": total / max(1, batches), **{f"val_{k}": v for k, v in val.items() if k != "examples"}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    test = evaluate(model, test_loader, criterion, device, idx_to_char, max_batches=args.max_eval_batches)
    summary = {
        "run_name": args.run_name,
        "device": str(device),
        "epochs": args.epochs,
        "image_width": args.image_width,
        "batch_size": args.batch_size,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "duration_sec": round(time.time() - started, 2),
        "test_loss": test["loss"],
        "test_cer": test["cer"],
        "test_wer": test["wer"],
        "test_batches": test["batches"],
        "examples": test["examples"],
        "history": history,
    }
    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)}, out_dir / "model.pt")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(test["examples"]).to_csv(out_dir / "prediction_examples.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="crnn_ctc_tiny")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
