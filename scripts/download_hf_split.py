from __future__ import annotations

import getpass
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "datasets" / "HF_BN-HTRd_Splitted"
REPO_ID = "shaoncsecu/BN-HTRd_Splitted"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    token = getpass.getpass("Hugging Face token: ").strip()
    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID)
    files = [s.rfilename for s in info.siblings]
    downloaded = {}
    for filename in ("README.md", "BN-HTRd_Split.zip"):
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            token=token,
            local_dir=DEST,
        )
        local = Path(path)
        if local.parent != DEST:
            target = DEST / filename
            shutil.copy2(local, target)
            local = target
        downloaded[filename] = {
            "path": str(local),
            "size": local.stat().st_size,
        }
    zip_path = DEST / "BN-HTRd_Split.zip"
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    summary = {
        "repo_id": REPO_ID,
        "sha": info.sha,
        "siblings": files,
        "downloaded": downloaded,
        "zip_members": len(infos),
        "zip_uncompressed_bytes": sum(i.file_size for i in infos),
        "zip_top_level": sorted({i.filename.split("/")[0] for i in infos if i.filename}),
    }
    (DEST / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
