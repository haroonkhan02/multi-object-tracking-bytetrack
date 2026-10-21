"""Download MOT17 from motchallenge.net and extract the training sequences.

The archive is 5.9 GB and holds train and test sets. Only the training sequences have public
ground truth, so only those are extracted. The three detector variants of each sequence share
the same images, so images and ground truth come from FRCNN and the other two variants keep
only their `det/det.txt`.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import (
    DETECTORS,
    IMAGE_DETECTOR,
    MOT17_DIR,
    MOT17_URL,
    MOT17_ZIP_BYTES,
    RAW_DIR,
    TRAIN_SEQUENCES,
)


def download_file(url: str, dest: Path, expected_bytes: int | None = None) -> Path:
    """Download with HTTP Range resume; safe to re-run after an interrupted transfer."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected_bytes and have == expected_bytes:
        return dest
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=60) as resp:
        resumed = resp.status == 206
        total = int(resp.headers.get("Content-Length", 0)) + (have if resumed else 0)
        mode = "ab" if resumed else "wb"
        with (
            dest.open(mode) as f,
            tqdm(total=total, initial=have if resumed else 0, unit="B", unit_scale=True) as bar,
        ):
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def wanted(member: str, sequences: tuple[str, ...] = TRAIN_SEQUENCES) -> bool:
    """Whether a zip member should be extracted."""
    parts = member.split("/")
    if len(parts) < 4 or parts[0] != "MOT17" or parts[1] != "train" or member.endswith("/"):
        return False
    folder, rest = parts[2], "/".join(parts[3:])
    seq, _, detector = folder.rpartition("-")
    if seq not in sequences or detector not in DETECTORS:
        return False
    if rest in ("det/det.txt", "seqinfo.ini"):
        return True
    return detector == IMAGE_DETECTOR and (rest == "gt/gt.txt" or rest.startswith("img1/"))


def extract_train(zip_path: Path, out_dir: Path = MOT17_DIR) -> int:
    """Extract the training sequences into `out_dir/train/...`; returns the files written."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if wanted(m)]
        for member in tqdm(members, desc="extracting", unit="file"):
            target = out_dir / Path(member).relative_to("MOT17")
            if target.exists() and target.stat().st_size == zf.getinfo(member).file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=RAW_DIR / "MOT17.zip")
    ap.add_argument("--skip-download", action="store_true", help="only extract an existing zip")
    ap.add_argument("--delete-zip", action="store_true", help="remove the archive afterwards")
    args = ap.parse_args(argv)

    if not args.skip_download:
        download_file(MOT17_URL, args.zip, MOT17_ZIP_BYTES)
    written = extract_train(args.zip)
    print(f"extracted {written} new files into {MOT17_DIR}")
    if args.delete_zip:
        args.zip.unlink()


if __name__ == "__main__":
    main()
