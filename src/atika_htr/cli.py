from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rich.console import Console
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets" / "BN-HTRd_Mendeley"
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"
SAMPLE_DIR = WORK / "Sample_Small"
console = Console()


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    expected_size: int
    sha256: str


ARCHIVES = [
    ArchiveSpec(
        "Automatic_Annotation.zip",
        1847761410,
        "dcb5a6760cc438aea4f84ad343cd44ce485c410f2d4c1851b5805aeebf11a7fc",
    ),
    ArchiveSpec(
        "BN-HTR_Dataset.zip",
        2608394039,
        "9ccb7e3e8ebfadc7b57d250496e12be59e1910b4759d150a909f1bffcef2a72e",
    ),
    ArchiveSpec(
        "Sample_Small.zip",
        76099297,
        "88afc8ce102a90b3d93c612de68c10b351faa788f2c330c8a8c7a882de8a115c",
    ),
    ArchiveSpec(
        "Structure_of_Directory.pdf",
        41039,
        "bafbfa75065e95b4f5fb917e5078c0ac3aa3da5fae5a24a1ca2d55ca7f21139f",
    ),
]


def ensure_dirs() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives() -> pd.DataFrame:
    rows = []
    for spec in ARCHIVES:
        path = DATASETS / spec.filename
        actual_size = path.stat().st_size if path.exists() else None
        actual_sha = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "file": spec.filename,
                "path": str(path),
                "expected_size": spec.expected_size,
                "actual_size": actual_size,
                "size_ok": actual_size == spec.expected_size,
                "expected_sha256": spec.sha256,
                "actual_sha256": actual_sha,
                "sha256_ok": actual_sha == spec.sha256,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "archive_verification.csv", index=False)
    return df


def extract_sample() -> Path:
    ensure_dirs()
    marker = SAMPLE_DIR / ".extracted"
    if marker.exists():
        return SAMPLE_DIR
    if SAMPLE_DIR.exists():
        shutil.rmtree(SAMPLE_DIR)
    with zipfile.ZipFile(DATASETS / "Sample_Small.zip") as zf:
        zf.extractall(WORK)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return SAMPLE_DIR


def archive_inventory() -> pd.DataFrame:
    rows = []
    for archive in DATASETS.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
        ext_counts = Counter(Path(info.filename).suffix.lower() or "<dir>" for info in infos)
        rows.append(
            {
                "archive": archive.name,
                "members": len(infos),
                "uncompressed_bytes": sum(info.file_size for info in infos),
                "jpg": ext_counts[".jpg"],
                "txt": ext_counts[".txt"],
                "xlsx": ext_counts[".xlsx"],
                "xml": ext_counts[".xml"],
                "pdf": ext_counts[".pdf"],
                "directories": ext_counts["<dir>"],
            }
        )
    df = pd.DataFrame(rows).sort_values("archive")
    df.to_csv(RESULTS / "archive_inventory.csv", index=False)
    return df


def sample_files() -> list[Path]:
    return [p for p in SAMPLE_DIR.rglob("*") if p.is_file()]


def line_text_files() -> list[Path]:
    return sorted((SAMPLE_DIR / "Segmentation_Images" / "Lines").rglob("*.txt"))


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(errors="replace")


def text_stats() -> pd.DataFrame:
    rows = []
    for path in sorted((SAMPLE_DIR / "Recognition_Ground_Truth_Texts").rglob("*.txt")):
        text = read_text(path).strip()
        words = text.split()
        chars = [c for c in text if not c.isspace()]
        rows.append(
            {
                "doc_id": path.stem,
                "file": str(path.relative_to(ROOT)),
                "characters_no_space": len(chars),
                "words": len(words),
                "lines": len([ln for ln in text.splitlines() if ln.strip()]),
                "unique_chars": len(set(chars)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sample_text_stats.csv", index=False)
    return df


def image_metrics(path: Path) -> dict[str, float | str | int]:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    lap_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    mean = float(image.mean())
    std = float(image.std())
    dark_fraction = float((image < 80).mean())
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = 255 - binary
    ink_fraction = float((foreground > 0).mean())
    return {
        "file": str(path.relative_to(ROOT)),
        "width": w,
        "height": h,
        "aspect_ratio": w / h if h else math.nan,
        "mean_intensity": mean,
        "std_intensity": std,
        "laplacian_var": lap_var,
        "dark_fraction": dark_fraction,
        "otsu_ink_fraction": ink_fraction,
    }


def collect_image_metrics(limit: int | None = None) -> pd.DataFrame:
    image_paths = sorted((SAMPLE_DIR / "Segmentation_Images" / "Lines").rglob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]
    rows = [image_metrics(path) for path in tqdm(image_paths, desc="image metrics")]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sample_image_quality_metrics.csv", index=False)
    return df


def preprocess_image(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return binary


def preprocessing_experiment(limit: int = 120) -> pd.DataFrame:
    out_dir = RESULTS / "preprocessed_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted((SAMPLE_DIR / "Segmentation_Images" / "Lines").rglob("*.jpg"))[:limit]
    rows = []
    for i, path in enumerate(tqdm(image_paths, desc="preprocess")):
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        processed = preprocess_image(raw)
        stem = f"{i:04d}_{path.stem}"
        if i < 12:
            cv2.imwrite(str(out_dir / f"{stem}.png"), processed)
        raw_lap = float(cv2.Laplacian(raw, cv2.CV_64F).var())
        proc_lap = float(cv2.Laplacian(processed, cv2.CV_64F).var())
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "raw_mean": float(raw.mean()),
                "processed_mean": float(processed.mean()),
                "raw_laplacian_var": raw_lap,
                "processed_laplacian_var": proc_lap,
                "raw_ink_fraction": float((raw < 180).mean()),
                "processed_ink_fraction": float((processed < 128).mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "preprocessing_comparison.csv", index=False)
    return df


def build_manifest() -> pd.DataFrame:
    rows = []
    for path in sample_files():
        rel = path.relative_to(SAMPLE_DIR)
        rows.append(
            {
                "sample_path": str(rel),
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "category": rel.parts[0] if rel.parts else "",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sample_manifest.csv", index=False)
    return df


def split_plan() -> pd.DataFrame:
    docs = sorted({p.parts[-2] for p in (SAMPLE_DIR / "Segmentation_Images" / "Lines").glob("*") if p.is_dir()})
    rows = []
    for i, doc_id in enumerate(docs):
        split = "test" if i % 5 == 0 else "val" if i % 5 == 1 else "train"
        rows.append({"doc_id": doc_id, "writer_proxy": doc_id, "split": split})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sample_writer_safe_split_plan.csv", index=False)
    return df


def plot_results(metrics: pd.DataFrame, prep: pd.DataFrame) -> None:
    if metrics.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(metrics["width"], bins=25, color="#2f6f6f")
    axes[0].set_title("Sample line image widths")
    axes[0].set_xlabel("pixels")
    axes[0].set_ylabel("count")
    axes[1].hist(metrics["otsu_ink_fraction"], bins=25, color="#9a5b2f")
    axes[1].set_title("Estimated ink fraction")
    axes[1].set_xlabel("fraction")
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_image_distributions.png", dpi=180)
    plt.close(fig)

    if not prep.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(prep["raw_ink_fraction"], prep["processed_ink_fraction"], alpha=0.7, color="#345995")
        ax.set_xlabel("raw estimated ink fraction")
        ax.set_ylabel("processed ink fraction")
        ax.set_title("Preprocessing shifts binarized ink density")
        fig.tight_layout()
        fig.savefig(RESULTS / "preprocessing_ink_shift.png", dpi=180)
        plt.close(fig)


def summarize_series(series: pd.Series) -> dict[str, float]:
    values = [float(v) for v in series.dropna()]
    if not values:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def make_report(
    verification: pd.DataFrame,
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    texts: pd.DataFrame,
    metrics: pd.DataFrame,
    prep: pd.DataFrame,
    splits: pd.DataFrame,
) -> Path:
    summary = {
        "archives_verified": bool(verification["sha256_ok"].all()),
        "sample_files": int(len(manifest)),
        "sample_jpg": int((manifest["extension"] == ".jpg").sum()),
        "sample_txt": int((manifest["extension"] == ".txt").sum()),
        "sample_xml": int((manifest["extension"] == ".xml").sum()),
        "sample_docs": int(texts["doc_id"].nunique()) if not texts.empty else 0,
        "sample_text_words": int(texts["words"].sum()) if not texts.empty else 0,
        "line_images_profiled": int(len(metrics)),
        "preprocessed_images": int(len(prep)),
        "image_width": summarize_series(metrics["width"]) if not metrics.empty else {},
        "image_height": summarize_series(metrics["height"]) if not metrics.empty else {},
        "ink_fraction": summarize_series(metrics["otsu_ink_fraction"]) if not metrics.empty else {},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = RESULTS / "thesis_results_report.md"
    lines = [
        "# Bangla HTR Robustness Thesis: Computed Setup and Preliminary Results",
        "",
        "## Environment",
        "",
        "- Platform: Apple Silicon macOS arm64",
        "- Python: uv-managed CPython 3.11",
        "- Package manager: uv",
        "",
        "## Dataset Status",
        "",
        f"- All Mendeley archives verified by SHA-256: **{summary['archives_verified']}**",
        f"- Downloaded archive folder: `{DATASETS}`",
        "- Hugging Face `shaoncsecu/BN-HTRd_Splitted` is access-restricted; an HF token is required before using it directly.",
        "",
        "### Archive Inventory",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Sample Dataset Findings",
        "",
        f"- Files extracted/profiled from `Sample_Small.zip`: **{summary['sample_files']}**",
        f"- Sample JPEG images: **{summary['sample_jpg']}**",
        f"- Sample text files: **{summary['sample_txt']}**",
        f"- Ground-truth documents in sample: **{summary['sample_docs']}**",
        f"- Ground-truth words in sample documents: **{summary['sample_text_words']}**",
        f"- Line/word images profiled: **{summary['line_images_profiled']}**",
        "",
        "### Image Quality Summary",
        "",
        f"- Width mean/median/min/max: {summary['image_width']}",
        f"- Height mean/median/min/max: {summary['image_height']}",
        f"- Otsu ink-fraction mean/median/min/max: {summary['ink_fraction']}",
        "",
        "## Preliminary Thesis Interpretation",
        "",
        "The proposal is technically viable, but the thesis should be framed as a controlled robustness study. "
        "The strongest design is to train and tune only on BN-HTRd, then evaluate on a separately annotated real-world subset.",
        "",
        "Recommended experiment ladder:",
        "",
        "1. Dataset inventory and quality profiling, already implemented here.",
        "2. Writer-safe BN-HTRd split construction, avoiding page or writer leakage.",
        "3. Baseline CRNN-CTC or transformer fine-tuning on line images.",
        "4. External evaluation on 300-500 manually annotated real-world line images.",
        "5. Error analysis by quality bucket: skew, noise, lighting, handwriting density, and segmentation defects.",
        "",
        "## Generated Files",
        "",
        "- `archive_verification.csv`",
        "- `archive_inventory.csv`",
        "- `sample_manifest.csv`",
        "- `sample_text_stats.csv`",
        "- `sample_image_quality_metrics.csv`",
        "- `preprocessing_comparison.csv`",
        "- `sample_writer_safe_split_plan.csv`",
        "- `sample_image_distributions.png`",
        "- `preprocessing_ink_shift.png`",
        "",
        "## Important Limitation",
        "",
        "These are preliminary computed results, not final OCR accuracy results. CER/WER requires running an OCR model "
        "and comparing predictions to line-level ground truth. The restricted Hugging Face split or a local full extraction "
        "plus label conversion is the next dependency for full model training.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    console.print("[bold]Verifying archives[/bold]")
    verification = verify_archives()
    console.print("[bold]Extracting sample dataset[/bold]")
    extract_sample()
    console.print("[bold]Building inventories and metrics[/bold]")
    inventory = archive_inventory()
    manifest = build_manifest()
    texts = text_stats()
    metrics = collect_image_metrics(limit=args.limit)
    prep = preprocessing_experiment(limit=min(args.limit or 120, 120))
    splits = split_plan()
    plot_results(metrics, prep)
    report = make_report(verification, inventory, manifest, texts, metrics, prep, splits)
    console.print(f"[green]Done.[/green] Report: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="atika-htr")
    sub = parser.add_subparsers(dest="command", required=True)
    all_parser = sub.add_parser("all", help="Run full local analysis pipeline")
    all_parser.add_argument("--limit", type=int, default=None, help="Limit images for faster profiling")
    verify_parser = sub.add_parser("verify", help="Verify downloaded archive checksums")
    extract_parser = sub.add_parser("extract-sample", help="Extract Sample_Small.zip")
    args = parser.parse_args()

    ensure_dirs()
    if args.command == "all":
        run_all(args)
    elif args.command == "verify":
        df = verify_archives()
        print(df.to_string(index=False))
    elif args.command == "extract-sample":
        print(extract_sample())


if __name__ == "__main__":
    main()
