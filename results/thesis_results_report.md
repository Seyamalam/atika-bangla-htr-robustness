# Bangla HTR Robustness Thesis: Current Local Results

This report summarizes the verified local BN-HTRd preparation and first in-domain OCR baseline. It does not yet include the external real-world robustness evaluation.

## Environment

- Platform: Apple Silicon macOS arm64
- Python: uv-managed CPython 3.11
- Package manager: uv
- OCR runtime: PyTorch CPU for CTC training because `CTCLoss` is not implemented on MPS in the current PyTorch build

## Dataset Status

- All Mendeley archives verified by SHA-256: **true**
- Downloaded archive folder: `/Users/seyam/Work/Research/Atika/datasets/BN-HTRd_Mendeley`
- Hugging Face `shaoncsecu/BN-HTRd_Splitted` metadata is visible, but the ZIP remains gated for the supplied account.

## Line-Level BN-HTRd Build

The verified local Mendeley archive was converted into line-level OCR manifests:

- `data/processed/bn_htrd_lines/labels.csv`
- `data/processed/bn_htrd_lines/train.csv`
- `data/processed/bn_htrd_lines/val.csv`
- `data/processed/bn_htrd_lines/test.csv`
- `data/processed/bn_htrd_lines/vocab.json`

Summary:

| Metric | Value |
|---|---:|
| Matched labeled line images | 14,113 |
| Source documents | 148 |
| Pages | 768 |
| Missing expected line images | 273 |
| Character vocabulary, including blank | 170 |
| Mean label length | 43.07 chars |
| Maximum label length | 86 chars |

Document-safe split:

| Split | Lines | Documents | Mean chars |
|---|---:|---:|---:|
| Train | 9,664 | 103 | 42.85 |
| Validation | 2,221 | 22 | 44.61 |
| Test | 2,228 | 23 | 42.49 |

## OCR Baselines

### Tiny CRNN-CTC Sanity Run

- Train/val/test rows: 300 / 100 / 100
- Epochs: 3
- Runtime: 11.7 seconds
- Test loss: 3.6465
- Test CER/WER: 100.00% / 100.00%

Interpretation: labels, image loading, Bangla vocabulary, batching, CTC loss, and decoding all execute correctly. Loss decreases, but greedy decoding remains blank at this tiny scale.

### Full BN-HTRd CRNN-CTC Baseline

- Train/val/test rows: 9,664 / 2,221 / 2,228
- Epochs: 5
- Runtime: 9.8 minutes
- Test loss: 1.0354
- Test CER: **27.76%**
- Test WER: **66.86%**

Validation trajectory:

| Epoch | Train loss | Val loss | Val CER | Val WER |
|---:|---:|---:|---:|---:|
| 1 | 3.4059 | 3.0355 | 82.62% | 98.42% |
| 2 | 2.5533 | 2.4023 | 59.75% | 96.39% |
| 3 | 1.7557 | 1.9050 | 48.50% | 87.35% |
| 4 | 1.2965 | 1.5900 | 40.44% | 79.37% |
| 5 | 1.0387 | 1.4414 | 37.67% | 75.52% |

Representative prediction:

| Truth | Prediction |
|---|---|
| বিশেষ সম্পাদকীয় | বিশেষ সমদকয় |
| চট্টগ্রামবাসীর পাশে দাঁড়াতে হবে বিগ বিজনেস | দটপ্রামবাসর পাশে দাড়াতে হবে কি বিজনেস |
| হাউসগুলোকে শুরুর দিকে কিছুটা ধীর গতিতে | হাউসগুলোকে শরুর দিকে কিুটা ধীর গতিতে |

## What Is Not Proven Yet

- No external real-world handwritten Bangla test set has been annotated or evaluated yet.
- The split is document-safe, not proven writer-safe.
- The CRNN-CTC baseline has not been tuned or trained to convergence.
- The model uses fixed-width resizing, greedy decoding, and Unicode codepoint labels.
- The result should not be compared directly with published BN-HTRd scores because this is a locally derived split.

## Next Experimental Step

The first three requested milestones are complete. Next, improve the in-domain baseline before using it as the thesis anchor:

1. Train longer with early stopping.
2. Add raw vs preprocessed image ablation.
3. Add dynamic-width batching or width buckets.
4. Annotate the external real-world line subset and report the robustness gap.
