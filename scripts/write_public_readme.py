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
    verification = pd.read_csv(RESULTS / "archive_verification.csv")
    text_stats = pd.read_csv(RESULTS / "sample_text_stats.csv")
    metrics = pd.read_csv(RESULTS / "sample_image_quality_metrics.csv")
    preprocessing = pd.read_csv(RESULTS / "preprocessing_comparison.csv")

    inv_md = inventory.assign(
        uncompressed_size=inventory["uncompressed_bytes"].map(lambda v: human_bytes(int(v)))
    )[["archive", "members", "uncompressed_size", "jpg", "txt", "xlsx", "xml", "pdf"]].to_markdown(index=False)
    verification_md = verification[["file", "actual_size", "size_ok", "sha256_ok"]].assign(
        actual_size=verification["actual_size"].map(lambda v: human_bytes(int(v)))
    ).to_markdown(index=False)
    text_md = text_stats.to_markdown(index=False)
    image_desc_md = metrics[
        [
            "width",
            "height",
            "aspect_ratio",
            "mean_intensity",
            "std_intensity",
            "laplacian_var",
            "dark_fraction",
            "otsu_ink_fraction",
        ]
    ].describe().round(3).to_markdown()
    prep_desc_md = preprocessing[
        [
            "raw_mean",
            "processed_mean",
            "raw_laplacian_var",
            "processed_laplacian_var",
            "raw_ink_fraction",
            "processed_ink_fraction",
        ]
    ].describe().round(3).to_markdown()
    prep_means = preprocessing[
        [
            "raw_laplacian_var",
            "processed_laplacian_var",
            "raw_ink_fraction",
            "processed_ink_fraction",
        ]
    ].mean().round(4).to_dict()

    readme = f"""# Cross-Dataset Robustness of Bangla Handwritten Text Recognition

This repository is a reproducible thesis/report workspace for evaluating how Bangla handwritten text recognition (HTR) systems behave when models trained on public benchmark data are tested against messier real-world handwriting.

The repository itself is the live manuscript: methods, computed results, plots, limitations, and next steps are all in this README. The implementation is designed for Apple Silicon using `uv` and a native `macos-aarch64` Python runtime.

## Status at a Glance

| Area | Current status |
|---|---|
| Dataset download | Official Mendeley BN-HTRd v4 archives downloaded locally |
| Integrity checks | All four downloaded Mendeley files pass size and SHA-256 checks |
| HF split | Metadata accessible, ZIP blocked by gated authorization |
| Sample extraction | `Sample_Small.zip` extracted and profiled |
| Image analysis | 359 line/word images profiled |
| Preprocessing analysis | 120 images processed and compared |
| OCR CER/WER | Not run yet; needs line-label conversion and model training |

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

### Archive Inventory

{inv_md}

### Download Verification

{verification_md}

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

### Ground-Truth Text Sample

The small public sample contains three ground-truth document text files. This confirms that the local workflow can read Bangla text metadata and produce document-level text statistics.

{text_md}

### Image Quality Metrics

The image profile table below is computed over 359 sample line/word JPEGs. `laplacian_var` is used as a simple sharpness/edge-detail proxy, while `otsu_ink_fraction` approximates foreground stroke density after Otsu thresholding.

{image_desc_md}

Interpretation:

- The median line/image width is **2048 px**, but heights vary widely, which means the sample mixes normal line images with taller crops or page-like fragments.
- The median estimated ink fraction is about **0.065**, so most images are sparse foreground on bright background.
- The max height of **3946 px** is a warning that training code must filter or normalize image aspect ratios before batching.

### Preprocessing Results

Preprocessing used denoising, CLAHE contrast enhancement, and adaptive thresholding. The goal here is not to claim OCR improvement yet; it is to measure how much the preprocessing changes image structure before model training.

{prep_desc_md}

Mean preprocessing shifts:

- Raw Laplacian variance: **{prep_means["raw_laplacian_var"]}**
- Processed Laplacian variance: **{prep_means["processed_laplacian_var"]}**
- Raw ink fraction: **{prep_means["raw_ink_fraction"]}**
- Processed ink fraction: **{prep_means["processed_ink_fraction"]}**

Interpretation:

- Edge/detail variance increases strongly after preprocessing, which is expected after binarization.
- Mean ink fraction stays close to the raw estimate, so preprocessing is not simply flooding the image with foreground.
- This preprocessing should be treated as an experimental condition, not a default: OCR models must be evaluated on raw and processed versions separately.

## Figures

![Sample image distributions](results/sample_image_distributions.png)

Figure 1. Sample image width and ink-density distributions. These measurements help identify whether the public benchmark contains layout and stroke-density variation that should be controlled during training and evaluation.

![Preprocessing ink shift](results/preprocessing_ink_shift.png)

Figure 2. Estimated ink-density shift after preprocessing. This is a first diagnostic for whether binarization/contrast normalization is changing image structure enough to affect recognition.

![Preprocessing preview grid](results/preprocessing_preview_grid.png)

Figure 3. Raw vs processed sample line images. This figure makes the preprocessing effect inspectable instead of only numeric.

## What We Should Do Next

The next useful work is not more packaging. It is to produce the first real OCR baseline and a valid evaluation protocol.

### Step 1: Build line-level labels from the local Mendeley archive

The Mendeley archives are available and verified. Since the Hugging Face split ZIP is gated, the practical route is to extract the full `BN-HTR_Dataset.zip`, parse the line-level XML/TXT structure, and create:

```text
data/processed/bn_htrd_lines/
├── images/
├── labels.csv
├── train.csv
├── val.csv
└── test.csv
```

The split must be writer/document safe. We should avoid putting line images from the same source document into both train and test.

### Step 2: Run a tiny sanity OCR experiment

Before training a large model, run a deliberately small baseline:

- 100-300 line images;
- character or grapheme vocabulary built from labels;
- CRNN + CTC;
- 1-3 quick epochs on Apple Silicon CPU/MPS if available;
- report whether loss decreases and whether decoding works.

This catches label/path/tokenization problems early.

### Step 3: Scale to the first thesis baseline

Once the tiny run works:

- train CRNN-CTC on the full line-level split;
- evaluate CER/WER on BN-HTRd test;
- rerun with preprocessing as a separate condition;
- save predictions, errors, and confusion examples.

### Step 4: Prepare the external real-world test set

For the 2703 real-world images, the most thesis-useful subset is:

- 300-500 manually annotated line images;
- stratified by clean/noisy/skewed/low-contrast/dense handwriting;
- never used for training;
- evaluated only after model choices are fixed.

### Step 5: Report the robustness gap

The core thesis result should be a table like:

| Model | Training data | Test data | Preprocessing | CER | WER | Robustness drop |
|---|---|---|---|---:|---:|---:|
| CRNN-CTC | BN-HTRd train | BN-HTRd test | Raw | TBD | TBD | baseline |
| CRNN-CTC | BN-HTRd train | External subset | Raw | TBD | TBD | TBD |
| CRNN-CTC | BN-HTRd train | BN-HTRd test | Processed | TBD | TBD | TBD |
| CRNN-CTC | BN-HTRd train | External subset | Processed | TBD | TBD | TBD |

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
