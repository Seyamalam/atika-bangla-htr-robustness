from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets" / "BN-HTRd_Mendeley" / "BN-HTR_Dataset.zip"
EXTRACT_ROOT = ROOT / "data" / "extracted" / "BN-HTR_Dataset"
OUT_ROOT = ROOT / "data" / "processed" / "bn_htrd_lines"
RESULTS = ROOT / "results"


def natural_key(value: str) -> tuple:
    out = []
    for part in value.replace(".", "_").split("_"):
        out.append(int(part) if part.isdigit() else part)
    return tuple(out)


def extract_needed(force: bool = False) -> Path:
    marker = EXTRACT_ROOT / ".selected_extracted"
    if marker.exists() and not force:
        return EXTRACT_ROOT / "BN-HTR_Dataset"
    if force and EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    prefixes = (
        "BN-HTR_Dataset/Recognition_Ground_Truth_Texts/",
        "BN-HTR_Dataset/Segmentation_Images/Lines/",
    )
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith(prefixes)]
        for member in tqdm(members, desc="extract BN-HTRd line data"):
            zf.extract(member, EXTRACT_ROOT)
    marker.write_text("ok\n", encoding="utf-8")
    return EXTRACT_ROOT / "BN-HTR_Dataset"


def read_word_table(path: Path) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(path)
        df = pd.DataFrame()
        for sheet in xls.sheet_names:
            candidate = pd.read_excel(path, sheet_name=sheet)
            if candidate.shape[1] >= 2 and len(candidate):
                df = candidate
                break
            candidate = pd.read_excel(path, sheet_name=sheet, header=None)
            if candidate.shape[1] >= 2 and len(candidate):
                df = candidate
                break
    except Exception:
        df = read_xlsx_raw_first_two_cols(path)
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(columns=["Id", "Word"])
    df.columns = [str(c).strip() for c in df.columns]
    if "Id" not in df.columns or "Word" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["Id", "Word"]
        df = df[df["Id"].astype(str).str.lower() != "id"]
    return df[["Id", "Word"]].dropna(subset=["Id", "Word"])


def read_xlsx_raw_first_two_cols(path: Path) -> pd.DataFrame:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def cell_col(ref: str) -> str:
        return "".join(ch for ch in ref if ch.isalpha())

    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared.append("".join(texts))
        sheet_names = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        for sheet_name in sorted(sheet_names):
            root = ET.fromstring(zf.read(sheet_name))
            rows = []
            for row in root.findall(".//main:row", ns):
                values = {}
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "")
                    col = cell_col(ref)
                    if col not in {"A", "B"}:
                        continue
                    typ = cell.attrib.get("t")
                    value = ""
                    if typ == "inlineStr":
                        texts = [t.text or "" for t in cell.findall(".//main:t", ns)]
                        value = "".join(texts)
                    else:
                        node = cell.find("main:v", ns)
                        if node is not None and node.text is not None:
                            value = node.text
                            if typ == "s":
                                idx = int(value)
                                value = shared[idx] if idx < len(shared) else ""
                    values[col] = value
                if values:
                    rows.append([values.get("A", ""), values.get("B", "")])
            if rows:
                return pd.DataFrame(rows, columns=["Id", "Word"])
    return pd.DataFrame(columns=["Id", "Word"])


def line_labels_from_xlsx(xlsx: Path) -> dict[str, str]:
    rows = read_word_table(xlsx)
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for _, row in rows.iterrows():
        word_id = str(row["Id"]).strip()
        word = str(row["Word"]).strip()
        parts = word_id.split("_")
        if len(parts) < 4 or not parts[-1].isdigit():
            continue
        line_id = "_".join(parts[:-1])
        grouped[line_id].append((int(parts[-1]), word))
    labels = {}
    for line_id, words in grouped.items():
        text = " ".join(word for _, word in sorted(words))
        text = " ".join(text.split())
        if text:
            labels[line_id] = text
    return labels


def image_shape(path: Path) -> tuple[int, int]:
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return 0, 0
    h, w = arr.shape[:2]
    return w, h


def build_manifest(force_extract: bool = False) -> pd.DataFrame:
    root = extract_needed(force=force_extract)
    gt_root = root / "Recognition_Ground_Truth_Texts"
    lines_root = root / "Segmentation_Images" / "Lines"
    rows = []
    missing = []
    for xlsx in tqdm(sorted(gt_root.glob("*/*.xlsx"), key=lambda p: natural_key(p.stem)), desc="build labels"):
        doc_id = xlsx.stem
        labels = line_labels_from_xlsx(xlsx)
        for line_id, text in labels.items():
            parts = line_id.split("_")
            if len(parts) < 3:
                continue
            page_id = "_".join(parts[:2])
            image_path = lines_root / doc_id / page_id / f"{line_id}.jpg"
            if not image_path.exists():
                missing.append({"doc_id": doc_id, "line_id": line_id, "expected_path": str(image_path)})
                continue
            width, height = image_shape(image_path)
            rows.append(
                {
                    "line_id": line_id,
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "image_path": str(image_path.relative_to(ROOT)),
                    "text": text,
                    "n_chars": len(text),
                    "n_words": len(text.split()),
                    "width": width,
                    "height": height,
                }
            )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["doc_id", "page_id", "line_id"], key=lambda col: col.map(natural_key))
    df.to_csv(OUT_ROOT / "labels.csv", index=False)
    pd.DataFrame(missing).to_csv(OUT_ROOT / "missing_line_images.csv", index=False)

    docs = sorted(df["doc_id"].unique(), key=natural_key)
    random.Random(42).shuffle(docs)
    n = len(docs)
    train_docs = set(docs[: int(n * 0.70)])
    val_docs = set(docs[int(n * 0.70) : int(n * 0.85)])
    test_docs = set(docs[int(n * 0.85) :])
    split_frames = {
        "train": df[df["doc_id"].isin(train_docs)],
        "val": df[df["doc_id"].isin(val_docs)],
        "test": df[df["doc_id"].isin(test_docs)],
    }
    for name, frame in split_frames.items():
        frame.to_csv(OUT_ROOT / f"{name}.csv", index=False)

    chars = sorted(set("".join(df["text"].tolist())))
    vocab = {"<blank>": 0}
    vocab.update({ch: i + 1 for i, ch in enumerate(chars)})
    (OUT_ROOT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "labels": int(len(df)),
        "documents": int(df["doc_id"].nunique()),
        "pages": int(df["page_id"].nunique()),
        "missing_line_images": int(len(missing)),
        "vocab_size_with_blank": int(len(vocab)),
        "max_chars": int(df["n_chars"].max()),
        "mean_chars": float(df["n_chars"].mean()),
        "splits": {
            name: {
                "rows": int(len(frame)),
                "documents": int(frame["doc_id"].nunique()),
                "mean_chars": float(frame["n_chars"].mean()) if len(frame) else 0.0,
            }
            for name, frame in split_frames.items()
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "line_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.describe(include="all").to_csv(RESULTS / "line_dataset_describe.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    build_manifest(force_extract=args.force_extract)


if __name__ == "__main__":
    main()
