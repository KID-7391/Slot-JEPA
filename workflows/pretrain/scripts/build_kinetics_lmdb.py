#!/usr/bin/env python3
"""Convert Kinetics videos into the LMDB format used by Slot-JEPA pretraining."""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
import os
import pickle
import sys
from typing import Iterable

import cv2
import decord
import lmdb
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def canonical_video_name(path: Path) -> str:
    """Match the canonicalization used by KineticsLMDBDataset."""
    name = path.stem.strip()
    return name[2:] if name.startswith("--") else name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample Kinetics videos at a fixed frame rate and store JPEG frames "
            "in the LMDB layout consumed by Slot-JEPA."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing one Kinetics split; searched recursively for videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output LMDB directory, for example data/Kinetics-400/kinetics_train.lmdb.",
    )
    parser.add_argument(
        "--valid-list",
        type=Path,
        default=None,
        help="Optional whitelist containing video basenames, with or without extensions.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frame sampling rate written to LMDB (default: 2.0).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=7,
        help="Skip videos yielding fewer sampled frames than a training clip (default: 7).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG encoding quality in [1, 100] (default: 95).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 1) // 2)),
        help="Parallel video decoding workers (default: half the CPUs, capped at 16).",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=512,
        help="LMDB virtual map size in GiB; sparse and not preallocated (default: 512).",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit and checkpoint metadata after this many videos (default: 100).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing conversion and skip videos recorded in metadata.pkl.",
    )
    return parser.parse_args()


def load_whitelist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Whitelist not found: {path}")
    names = {
        canonical_video_name(Path(line.strip()))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not names:
        raise ValueError(f"Whitelist is empty: {path}")
    return names


def discover_videos(root: Path, whitelist: set[str] | None) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if whitelist is not None:
        videos = [path for path in videos if canonical_video_name(path) in whitelist]
    if not videos:
        raise RuntimeError(f"No matching videos found under {root}")

    seen: dict[str, Path] = {}
    for path in videos:
        name = canonical_video_name(path)
        try:
            name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Non-ASCII video basename is unsupported: {path}") from exc
        if name in seen:
            raise ValueError(
                "Duplicate canonical video basename; LMDB keys would collide: "
                f"{seen[name]} and {path}"
            )
        seen[name] = path
    return videos


def sample_indices(frame_count: int, source_fps: float, sample_fps: float) -> np.ndarray:
    if frame_count <= 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source FPS: {source_fps}")
    step = source_fps / sample_fps
    indices = np.floor(np.arange(0.0, frame_count, step)).astype(np.int64)
    return np.unique(np.clip(indices, 0, frame_count - 1))


def convert_video(task: tuple[str, float, int, int]) -> tuple[str, list[bytes] | None, str | None]:
    path_text, sample_fps, min_frames, jpeg_quality = task
    path = Path(path_text)
    name = canonical_video_name(path)
    try:
        cv2.setNumThreads(1)
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(path), num_threads=1)
        indices = sample_indices(len(reader), float(reader.get_avg_fps()), sample_fps)
        if len(indices) < min_frames:
            raise ValueError(f"only {len(indices)} sampled frames (minimum: {min_frames})")

        rgb_frames = reader.get_batch(indices.tolist()).asnumpy()
        encoded_frames: list[bytes] = []
        encode_options = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        for rgb_frame in rgb_frames:
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr_frame, encode_options)
            if not ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            encoded_frames.append(encoded.tobytes())
        return name, encoded_frames, None
    except Exception as exc:  # noqa: BLE001 - corrupted videos must not stop the full conversion
        return name, None, f"{path}: {type(exc).__name__}: {exc}"


def atomic_write_metadata(output: Path, metadata: dict[str, int]) -> None:
    ordered_metadata = dict(sorted(metadata.items()))
    metadata_tmp = output / "metadata.pkl.tmp"
    names_tmp = output / "valid_video_names.txt.tmp"
    with metadata_tmp.open("wb") as stream:
        pickle.dump(ordered_metadata, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    names_tmp.write_text(
        "".join(f"{name}\n" for name in ordered_metadata), encoding="utf-8"
    )
    metadata_tmp.replace(output / "metadata.pkl")
    names_tmp.replace(output / "valid_video_names.txt")


def load_existing_metadata(output: Path, resume: bool) -> dict[str, int]:
    metadata_path = output / "metadata.pkl"
    if not output.exists():
        return {}
    contents = list(output.iterdir())
    if not contents:
        return {}
    if not resume:
        raise FileExistsError(
            f"Output is not empty: {output}. Choose another path or pass --resume."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {metadata_path}")
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"Invalid metadata in {metadata_path}")
    return metadata


def tasks_for(
    videos: Iterable[Path], sample_fps: float, min_frames: int, jpeg_quality: int
) -> Iterable[tuple[str, float, int, int]]:
    for path in videos:
        yield str(path), sample_fps, min_frames, jpeg_quality


def main() -> int:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    if args.min_frames <= 0:
        raise ValueError("--min-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.workers <= 0 or args.commit_every <= 0 or args.map_size_gb <= 0:
        raise ValueError("--workers, --commit-every, and --map-size-gb must be positive")

    whitelist = load_whitelist(args.valid_list)
    videos = discover_videos(args.input, whitelist)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = load_existing_metadata(args.output, args.resume)
    pending = [path for path in videos if canonical_video_name(path) not in metadata]

    print(f"Discovered {len(videos):,} videos; {len(pending):,} remain to convert.")
    if not pending:
        atomic_write_metadata(args.output, metadata)
        return 0

    environment = lmdb.open(
        str(args.output),
        map_size=args.map_size_gb * 1024**3,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )
    transaction = environment.begin(write=True)
    converted_since_commit = 0
    converted_total = 0
    failures: list[str] = []

    try:
        with Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(
                convert_video,
                tasks_for(pending, args.sample_fps, args.min_frames, args.jpeg_quality),
                chunksize=1,
            )
            for completed, (name, frames, error) in enumerate(results, start=1):
                if error is not None or frames is None:
                    failures.append(error or f"{name}: unknown conversion failure")
                    print(f"[skip] {failures[-1]}", file=sys.stderr)
                    continue

                try:
                    for index, jpeg_bytes in enumerate(frames):
                        key = f"{name}_frame_{index:06d}".encode("ascii")
                        transaction.put(key, jpeg_bytes, overwrite=True)
                except lmdb.MapFullError as exc:
                    transaction.abort()
                    raise RuntimeError(
                        "LMDB map is full; rerun with --resume and a larger --map-size-gb"
                    ) from exc

                metadata[name] = len(frames)
                converted_total += 1
                converted_since_commit += 1
                if converted_since_commit >= args.commit_every:
                    transaction.commit()
                    environment.sync()
                    atomic_write_metadata(args.output, metadata)
                    transaction = environment.begin(write=True)
                    converted_since_commit = 0
                if completed % 100 == 0 or completed == len(pending):
                    print(
                        f"Processed {completed:,}/{len(pending):,}; "
                        f"stored {converted_total:,}; skipped {len(failures):,}"
                    )

        transaction.commit()
        environment.sync()
        atomic_write_metadata(args.output, metadata)
    except Exception:
        try:
            transaction.abort()
        except lmdb.Error:
            pass
        raise
    finally:
        environment.close()

    if failures:
        failure_log = args.output / "failed_videos.txt"
        failure_log.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Skipped {len(failures):,} videos; details: {failure_log}", file=sys.stderr)
    print(f"LMDB ready: {args.output} ({len(metadata):,} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
