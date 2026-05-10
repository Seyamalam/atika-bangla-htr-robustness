# Bangla HTR Robustness Thesis: Computed Setup and Preliminary Results

## Environment

- Platform: Apple Silicon macOS arm64
- Python: uv-managed CPython 3.11
- Package manager: uv

## Dataset Status

- All Mendeley archives verified by SHA-256: **True**
- Downloaded archive folder: `/Users/seyam/Work/Research/Atika/datasets/BN-HTRd_Mendeley`
- Hugging Face `shaoncsecu/BN-HTRd_Splitted` is access-restricted; an HF token is required before using it directly.

### Archive Inventory

| archive                  |   members |   uncompressed_bytes |    jpg |   txt |   xlsx |   xml |   pdf |   directories |
|:-------------------------|----------:|---------------------:|-------:|------:|-------:|------:|------:|--------------:|
| Automatic_Annotation.zip |    154374 |           1918135594 | 121572 | 15694 |      0 |     0 |    61 |         17020 |
| BN-HTR_Dataset.zip       |    185856 |           2729377734 | 137721 | 16256 |    150 | 15168 |   127 |         16410 |
| Sample_Small.zip         |      4378 |             81108514 |   3225 |   387 |      3 |   360 |     3 |           394 |

## Sample Dataset Findings

- Files extracted/profiled from `Sample_Small.zip`: **3985**
- Sample JPEG images: **3225**
- Sample text files: **387**
- Ground-truth documents in sample: **3**
- Ground-truth words in sample documents: **2363**
- Line/word images profiled: **359**

### Image Quality Summary

- Width mean/median/min/max: {'mean': 1893.8217270194987, 'median': 2048.0, 'min': 313.0, 'max': 2448.0}
- Height mean/median/min/max: {'mean': 408.5877437325905, 'median': 217.0, 'min': 81.0, 'max': 3946.0}
- Otsu ink-fraction mean/median/min/max: {'mean': 0.06924565867286134, 'median': 0.06525888166947864, 'min': 0.011912828279900284, 'max': 0.16321237061977803}

## Preliminary Thesis Interpretation

The proposal is technically viable, but the thesis should be framed as a controlled robustness study. The strongest design is to train and tune only on BN-HTRd, then evaluate on a separately annotated real-world subset.

Recommended experiment ladder:

1. Dataset inventory and quality profiling, already implemented here.
2. Writer-safe BN-HTRd split construction, avoiding page or writer leakage.
3. Baseline CRNN-CTC or transformer fine-tuning on line images.
4. External evaluation on 300-500 manually annotated real-world line images.
5. Error analysis by quality bucket: skew, noise, lighting, handwriting density, and segmentation defects.

## Generated Files

- `archive_verification.csv`
- `archive_inventory.csv`
- `sample_manifest.csv`
- `sample_text_stats.csv`
- `sample_image_quality_metrics.csv`
- `preprocessing_comparison.csv`
- `sample_writer_safe_split_plan.csv`
- `sample_image_distributions.png`
- `preprocessing_ink_shift.png`

## Important Limitation

These are preliminary computed results, not final OCR accuracy results. CER/WER requires running an OCR model and comparing predictions to line-level ground truth. The restricted Hugging Face split or a local full extraction plus label conversion is the next dependency for full model training.
