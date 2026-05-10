from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    hf_status = json.loads((RESULTS / "hf_access_status.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(RESULTS / "archive_inventory.csv")

    inv_md = inventory.assign(
        uncompressed_size=inventory["uncompressed_bytes"].map(lambda v: human_bytes(int(v)))
    )[["archive", "members", "uncompressed_size", "jpg", "txt", "xlsx", "xml", "pdf"]].to_markdown(index=False)

    readme = f"""# Cross-Dataset Robustness of Bangla Handwritten Text Recognition

This repository is a reproducible thesis/report workspace for evaluating how Bangla handwritten text recognition (HTR) systems behave when models trained on public benchmark data are tested against messier real-world handwriting.

The current implementation focuses on dataset acquisition, verification, inventory, image-quality analysis, preprocessing experiments, split planning, and manuscript-ready reporting. It is designed for Apple Silicon using `uv` and a native `macos-aarch64` Python runtime.

## Research Question

How robust are Bangla HTR models trained on BN-HTRd when evaluated outside their original benchmark distribution?

The thesis contribution is framed around robustness rather than simply building another OCR model:

- quantify the dataset gap between BN-HTRd and real-world handwritten Bangla images;
- control the training/evaluation protocol so external performance is meaningful;
- analyze which image-quality factors are likely to drive recognition failure;
- prepare an annotation and evaluation workflow for a 300-500 line real-world external test set.

## Methodology

```mermaid
flowchart LR
    A[BN-HTRd public benchmark] --> B[Archive verification]
    B --> C[Dataset inventory]
    C --> D[Line/image quality profiling]
    D --> E[Preprocessing experiment]
    E --> F[Writer-safe split plan]
    F --> G[HTR model training]
    G --> H[BN-HTRd CER/WER]
    G --> I[External real-world CER/WER]
    I --> J[Robustness gap and error analysis]
```

```mermaid
flowchart TD
    R[Raw handwritten image] --> Q{{Quality factors}}
    Q --> S[Skew]
    Q --> N[Noise and lighting]
    Q --> D[Stroke density]
    Q --> L[Line/word segmentation]
    S --> M[OCR model]
    N --> M
    D --> M
    L --> M
    M --> E[CER/WER and failure buckets]
```

## Environment

- Machine: Apple Silicon (`arm64`)
- Python: uv-managed CPython `3.11.15`
- Package manager: `uv`
- Core libraries: OpenCV, Pillow, NumPy, pandas, matplotlib, Hugging Face Hub

Reproduce the current results:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
uv run atika-htr all
```

## Dataset Status

The official Mendeley BN-HTRd v4 files were downloaded and verified by SHA-256. Raw datasets are intentionally excluded from GitHub because they are large and should be retrieved from the original source.

{inv_md}

Hugging Face split status:

- Repository: `shaoncsecu/BN-HTRd_Splitted`
- Status: `{hf_status["status"]}`
- Reason: {hf_status["reason"]}
- Next action: {hf_status["next_action"]}

## Computed Preliminary Results

From the extracted `Sample_Small.zip` subset:

- Files extracted/profiled: **{summary["sample_files"]:,}**
- JPEG images: **{summary["sample_jpg"]:,}**
- Text files: **{summary["sample_txt"]:,}**
- XML annotation files: **{summary["sample_xml"]:,}**
- Ground-truth documents: **{summary["sample_docs"]:,}**
- Ground-truth words: **{summary["sample_text_words"]:,}**
- Line/word images profiled: **{summary["line_images_profiled"]:,}**
- Images preprocessed for comparison: **{summary["preprocessed_images"]:,}**

Image profile summary:

- Width mean/median/min/max: `{summary["image_width"]}`
- Height mean/median/min/max: `{summary["image_height"]}`
- Otsu ink-fraction mean/median/min/max: `{summary["ink_fraction"]}`

## Figures

![Sample image distributions](results/sample_image_distributions.png)

Figure 1. Sample image width and ink-density distributions. These measurements help identify whether the public benchmark contains layout and stroke-density variation that should be controlled during training and evaluation.

![Preprocessing ink shift](results/preprocessing_ink_shift.png)

Figure 2. Estimated ink-density shift after preprocessing. This is a first diagnostic for whether binarization/contrast normalization is changing image structure enough to affect recognition.

## Methodological Notes

The current run produces preliminary dataset and preprocessing results, not final OCR accuracy. Final CER/WER requires a trained OCR model and clean line-level ground truth. The recommended experimental ladder is:

1. Create writer-safe train/validation/test splits for BN-HTRd.
2. Train a compact baseline model such as CRNN-CTC on line-level images.
3. Fine-tune one stronger transformer or grapheme-tokenized model.
4. Annotate 300-500 external real-world line images.
5. Report the in-domain BN-HTRd CER/WER, external CER/WER, and robustness drop.
6. Break errors down by quality bucket: skew, noise, lighting, stroke density, and segmentation defects.

## Repository Layout

```text
src/atika_htr/cli.py                 Reproducible analysis CLI
scripts/download_hf_split.py         HF gated split downloader, token read from stdin
scripts/write_public_readme.py       README manuscript generator
scripts/build_manuscript_docx.py     DOCX manuscript generator
results/                            Generated result tables and figures
```

Raw data folders such as `datasets/` and `data/` are ignored by git.

## Citation Pointers

- BN-HTRd dataset DOI: `10.17632/743k6dm543.4`
- Original BN-HTRd paper: `arXiv:2206.08977`
- BN-DRISHTI model/demo repository: `crusnic-corp/BN-DRISHTI`

## Current Limitation

The Hugging Face `BN-HTRd_Splitted` archive is gated. The supplied token was not authorized for the ZIP, so this repository uses the verified public Mendeley archives and records the gated-access state in `results/hf_access_status.json`.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

