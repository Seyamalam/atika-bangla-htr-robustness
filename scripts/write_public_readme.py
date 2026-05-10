from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_ocr_history(run_dir: Path, output: Path, title: str) -> None:
    history = pd.read_csv(run_dir / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title("CTC loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_cer"], marker="o", label="Validation CER")
    axes[1].plot(history["epoch"], history["val_wer"], marker="o", label="Validation WER")
    axes[1].set_title("Validation error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    hf_status = json.loads((RESULTS / "hf_access_status.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(RESULTS / "archive_inventory.csv")
    verification = pd.read_csv(RESULTS / "archive_verification.csv")
    text_stats = pd.read_csv(RESULTS / "sample_text_stats.csv")
    metrics = pd.read_csv(RESULTS / "sample_image_quality_metrics.csv")
    preprocessing = pd.read_csv(RESULTS / "preprocessing_comparison.csv")
    line_summary = json.loads((RESULTS / "line_dataset_summary.json").read_text(encoding="utf-8"))
    tiny_summary = json.loads((RESULTS / "crnn_ctc_tiny" / "summary.json").read_text(encoding="utf-8"))
    full_summary = json.loads((RESULTS / "crnn_ctc_full_baseline" / "summary.json").read_text(encoding="utf-8"))
    full_history = pd.read_csv(RESULTS / "crnn_ctc_full_baseline" / "history.csv")
    tiny_history = pd.read_csv(RESULTS / "crnn_ctc_tiny" / "history.csv")
    full_examples = pd.read_csv(RESULTS / "crnn_ctc_full_baseline" / "prediction_examples.csv").head(8)

    plot_ocr_history(
        RESULTS / "crnn_ctc_tiny",
        RESULTS / "crnn_ctc_tiny_training.png",
        "Tiny CRNN-CTC Sanity Run",
    )
    plot_ocr_history(
        RESULTS / "crnn_ctc_full_baseline",
        RESULTS / "crnn_ctc_full_baseline_training.png",
        "Full BN-HTRd CRNN-CTC Baseline",
    )

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
    split_rows = [
        {
            "split": name,
            "rows": values["rows"],
            "documents": values["documents"],
            "mean_chars": round(values["mean_chars"], 2),
        }
        for name, values in line_summary["splits"].items()
    ]
    split_md = pd.DataFrame(split_rows).to_markdown(index=False)
    tiny_history_md = tiny_history.round(4).to_markdown(index=False)
    full_history_md = full_history.round(4).to_markdown(index=False)
    examples_md = full_examples.to_markdown(index=False)

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
| Line labels | Built from verified local Mendeley BN-HTRd archive |
| Tiny OCR sanity run | Completed on 300 training lines |
| First thesis OCR baseline | Completed on full local BN-HTRd line split |
| Current in-domain test result | CRNN-CTC raw-line baseline: **CER {pct(full_summary["test_cer"])}**, **WER {pct(full_summary["test_wer"])}** |

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
uv run python scripts/build_line_dataset.py
uv run python scripts/train_crnn_ctc.py --run-name crnn_ctc_tiny --epochs 3 --batch-size 8 --image-width 384 --max-train-rows 300 --max-val-rows 100 --max-test-rows 100 --device cpu
uv run python scripts/train_crnn_ctc.py --run-name crnn_ctc_full_baseline --epochs 5 --batch-size 16 --image-width 384 --device cpu
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

## Line-Level Dataset Build

The first requested blocker is now resolved. The workflow extracts only the needed directories from the verified Mendeley archive, reads the recognition ground-truth spreadsheets, reconstructs line text from word-level records, matches each line to its JPEG crop, and writes document-safe train/validation/test splits.

Generated local files:

```text
data/processed/bn_htrd_lines/
├── labels.csv
├── train.csv
├── val.csv
├── test.csv
├── vocab.json
├── missing_line_images.csv
└── summary.json
```

Dataset summary:

| Metric | Value |
|---|---:|
| Matched labeled line images | {line_summary["labels"]:,} |
| Source documents | {line_summary["documents"]:,} |
| Pages | {line_summary["pages"]:,} |
| Missing expected line images | {line_summary["missing_line_images"]:,} |
| Character vocabulary, including CTC blank | {line_summary["vocab_size_with_blank"]:,} |
| Maximum label length | {line_summary["max_chars"]:,} chars |
| Mean label length | {line_summary["mean_chars"]:.2f} chars |

Split summary:

{split_md}

The split is document-safe, so lines from the same source document do not appear across train, validation, and test. The 273 missing line images are recorded in `data/processed/bn_htrd_lines/missing_line_images.csv` and excluded from training/evaluation.

## OCR Experiments

The OCR model is a compact CRNN-CTC baseline: four convolution blocks, a two-layer bidirectional LSTM, character-level CTC output, greedy CTC decoding, and CER/WER evaluation. PyTorch detects Apple Silicon MPS on this machine, but `CTCLoss` is not implemented for MPS in the current PyTorch build, so these CTC runs use CPU for correctness and reproducibility.

### Tiny Sanity Baseline

Command:

```bash
uv run python scripts/train_crnn_ctc.py --run-name crnn_ctc_tiny --epochs 3 --batch-size 8 --image-width 384 --max-train-rows 300 --max-val-rows 100 --max-test-rows 100 --device cpu
```

Results:

| Metric | Value |
|---|---:|
| Training lines | {tiny_summary["train_rows"]:,} |
| Validation lines | {tiny_summary["val_rows"]:,} |
| Test lines | {tiny_summary["test_rows"]:,} |
| Runtime | {tiny_summary["duration_sec"]:.1f} sec |
| Test CTC loss | {tiny_summary["test_loss"]:.4f} |
| Test CER | {pct(tiny_summary["test_cer"])} |
| Test WER | {pct(tiny_summary["test_wer"])} |

Training history:

{tiny_history_md}

Interpretation: this deliberately tiny 300-line, 3-epoch run verifies that labels load, line images batch correctly, the Bangla character vocabulary is usable, and CTC loss decreases. It still decodes blank strings on the test examples, so it is a wiring sanity check, not an accuracy result.

### Full BN-HTRd Baseline

Command:

```bash
uv run python scripts/train_crnn_ctc.py --run-name crnn_ctc_full_baseline --epochs 5 --batch-size 16 --image-width 384 --device cpu
```

Results:

| Metric | Value |
|---|---:|
| Training lines | {full_summary["train_rows"]:,} |
| Validation lines | {full_summary["val_rows"]:,} |
| Test lines | {full_summary["test_rows"]:,} |
| Runtime | {full_summary["duration_sec"] / 60:.1f} min |
| Test CTC loss | {full_summary["test_loss"]:.4f} |
| Test CER | **{pct(full_summary["test_cer"])}** |
| Test WER | **{pct(full_summary["test_wer"])}** |

Training history:

{full_history_md}

Representative predictions:

{examples_md}

Interpretation: the full run learns a real recognizer and produces non-blank Bangla predictions. Validation CER improves from **{pct(full_summary["history"][0]["val_cer"])}** to **{pct(full_summary["history"][-1]["val_cer"])}** over five epochs, and validation loss is still falling at epoch 5. This is a first thesis baseline, not a final model; longer training, better width handling, stronger augmentations, and a preprocessing condition should all be evaluated next.

## Figures

![Sample image distributions](results/sample_image_distributions.png)

Figure 1. Sample image width and ink-density distributions. These measurements help identify whether the public benchmark contains layout and stroke-density variation that should be controlled during training and evaluation.

![Preprocessing ink shift](results/preprocessing_ink_shift.png)

Figure 2. Estimated ink-density shift after preprocessing. This is a first diagnostic for whether binarization/contrast normalization is changing image structure enough to affect recognition.

![Preprocessing preview grid](results/preprocessing_preview_grid.png)

Figure 3. Raw vs processed sample line images. This figure makes the preprocessing effect inspectable instead of only numeric.

![Tiny CRNN-CTC training curves](results/crnn_ctc_tiny_training.png)

Figure 4. Tiny sanity-run CTC loss and validation CER/WER. Loss decreases, but greedy decoding remains blank after three short epochs.

![Full CRNN-CTC baseline training curves](results/crnn_ctc_full_baseline_training.png)

Figure 5. Full BN-HTRd CRNN-CTC baseline. Validation loss, CER, and WER improve across all five epochs.

## What We Should Do Next

The next useful work is to improve the baseline and add the external robustness test. The first three requested milestones are now complete: line labels/splits, a tiny OCR sanity run, and a full in-domain CRNN-CTC baseline.

### Step 4: Improve the in-domain baseline

- Train the CRNN-CTC baseline for more epochs with early stopping.
- Compare raw images against the preprocessing pipeline as a separate condition.
- Add width bucketing or dynamic-width batches so long lines are not compressed as aggressively.
- Add light geometric/contrast augmentation that matches real handwriting noise.

### Step 5: Prepare the external real-world test set

For the 2703 real-world images, the most thesis-useful subset is:

- 300-500 manually annotated line images;
- stratified by clean/noisy/skewed/low-contrast/dense handwriting;
- never used for training;
- evaluated only after model choices are fixed.

### Step 6: Report the robustness gap

The core thesis result should be a table like:

| Model | Training data | Test data | Preprocessing | CER | WER | Robustness drop |
|---|---|---|---|---:|---:|---:|
| CRNN-CTC | BN-HTRd train | BN-HTRd test | Raw | {pct(full_summary["test_cer"])} | {pct(full_summary["test_wer"])} | baseline |
| CRNN-CTC | BN-HTRd train | External subset | Raw | TBD | TBD | TBD |
| CRNN-CTC | BN-HTRd train | BN-HTRd test | Processed | TBD | TBD | TBD |
| CRNN-CTC | BN-HTRd train | External subset | Processed | TBD | TBD | TBD |

## Methodological Notes

The current run produces preliminary dataset, preprocessing, and first OCR baseline results. The recommended experimental ladder from here is:

1. Extend the CRNN-CTC run until validation CER plateaus.
2. Rerun the baseline with preprocessing and augmentation as controlled ablations.
3. Fine-tune one stronger transformer or grapheme-tokenized model.
4. Annotate 300-500 external real-world line images.
5. Report the in-domain BN-HTRd CER/WER, external CER/WER, and robustness drop.
6. Break errors down by quality bucket: skew, noise, lighting, stroke density, and segmentation defects.

## Repository Layout

```text
src/atika_htr/cli.py                 Reproducible analysis CLI
scripts/download_hf_split.py         HF gated split downloader, token read from stdin
scripts/build_line_dataset.py        BN-HTRd line label and split builder
scripts/train_crnn_ctc.py            CRNN-CTC sanity/full baseline trainer
scripts/write_public_readme.py       README manuscript generator
results/                            Generated result tables and figures
```

Raw data folders such as `datasets/` and `data/` are ignored by git.

## Citation Pointers

- BN-HTRd dataset DOI: `10.17632/743k6dm543.4`
- Original BN-HTRd paper: `arXiv:2206.08977`
- BN-DRISHTI model/demo repository: `crusnic-corp/BN-DRISHTI`

## Current Limitation

The Hugging Face `BN-HTRd_Splitted` archive is gated. The supplied token was not authorized for the ZIP, so this repository uses the verified public Mendeley archives and records the gated-access state in `results/hf_access_status.json`. Model checkpoints are excluded from GitHub; metrics, histories, prediction examples, scripts, and plots are published.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
