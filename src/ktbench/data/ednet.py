"""EdNet KT1 (Choi et al., 2020): 784,309 students' answers from the Santa app.

    python -m ktbench.data.ednet --sample 50000      # the benchmark sample
    python -m ktbench.data.ednet --sample 0          # every student

KT1 is one CSV per student (``u<id>.csv``) with the letter the student picked,
not right/wrong. The pipeline:

1. Reads each student's CSV straight out of the KT1 zip (no 5 GB extract), in
   parallel.
2. Joins every answer to ``questions.csv`` from the Contents zip for the
   correct letter and the skill tags; correct = (picked letter == correct letter).
3. Writes the joined rows as Parquet parts, one per block of students, under
   ``data/interim/ednet/<sample>/``, so a crash resumes where it stopped.
4. Sorts each student by timestamp, drops students with fewer than 3 answers
   and questions with no tags, and writes the canonical table.

License: CC BY-NC 4.0, research and non-commercial use only. Never ship a model
trained on EdNet inside a commercial product.
"""

from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ktbench.data import schema

NAME = "ednet"
KT1_ID = "1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw"  # bit.ly/ednet_kt1
CONTENTS_ID = "117aYJAWG3GU48suS66NPaB82HwFj6xWS"  # bit.ly/ednet-content
KT1_ZIP, CONTENTS_ZIP = "EdNet-KT1.zip", "EdNet-Contents.zip"
MIN_SEQ_LEN = 3
BLOCK = 2000


def _drive(file_id: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {path.name} ...")
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise RuntimeError(f"{path.name}: download is not a zip; fetch it by hand from the EdNet repo")
    tmp.rename(path)
    return path


def load_questions(contents_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(contents_zip) as z:
        name = next(n for n in z.namelist() if n.endswith("questions.csv") and "__MACOSX" not in n)
        with z.open(name) as f:
            q = pd.read_csv(f, usecols=["question_id", "correct_answer", "tags"], dtype=str)
    q["tags"] = q["tags"].fillna("").map(lambda s: [t for t in s.split(";") if t and t != "-1"])
    return q


def student_files(kt1_zip: Path) -> list[str]:
    with zipfile.ZipFile(kt1_zip) as z:
        return sorted(n for n in z.namelist() if n.endswith(".csv") and "/u" in "/" + n and "__MACOSX" not in n)


def _read_block(args: tuple[str, list[str]]) -> pd.DataFrame:
    kt1_zip, names = args
    frames = []
    with zipfile.ZipFile(kt1_zip) as z:
        for name in names:
            with z.open(name) as f:
                d = pd.read_csv(f, usecols=["timestamp", "question_id", "user_answer"], dtype={"user_answer": str})
            d["user"] = Path(name).stem
            d["_row"] = np.arange(len(d))
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(data_dir: Path, sample: int = 50_000, seed: int = 0, workers: int = 8) -> Path:
    raw = data_dir / "raw" / NAME
    kt1 = _drive(KT1_ID, raw / KT1_ZIP)
    questions = load_questions(_drive(CONTENTS_ID, raw / CONTENTS_ZIP)).set_index("question_id")

    names = student_files(kt1)
    tag = f"sample{sample}" if sample else "full"
    if sample and sample < len(names):
        rng = np.random.default_rng(seed)
        names = sorted(rng.choice(names, size=sample, replace=False).tolist())
    print(f"{NAME}: {len(names):,} students ({tag})")

    interim = data_dir / "interim" / NAME / tag
    interim.mkdir(parents=True, exist_ok=True)
    blocks = [(str(kt1), names[i : i + BLOCK]) for i in range(0, len(names), BLOCK)]
    todo = [(i, b) for i, b in enumerate(blocks) if not (interim / f"part-{i:05d}.parquet").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (i, _), d in zip(todo, pool.map(_read_block, [b for _, b in todo]), strict=True):
            d = d.join(questions, on="question_id")
            d["correct"] = (d["user_answer"].str.lower() == d["correct_answer"]).astype(np.int8)
            d[["user", "_row", "timestamp", "question_id", "correct", "tags"]].to_parquet(
                interim / f"part-{i:05d}.parquet", index=False
            )
            print(f"  block {i + 1}/{len(blocks)}: {len(d):,} answers")

    d = pd.concat([pd.read_parquet(p) for p in sorted(interim.glob("part-*.parquet"))], ignore_index=True)
    d = d[d["tags"].map(lambda t: t is not None and len(t) > 0)]
    d = d.sort_values(["user", "timestamp", "_row"], kind="stable")
    df = pd.DataFrame(
        {
            "user_id": d["user"].str.lstrip("u").astype(np.int64).to_numpy(),
            "item_id": d["question_id"].str.lstrip("q").astype(np.int64).to_numpy(),
            "skills": d["tags"].map(list).to_numpy(),
            "correct": d["correct"].to_numpy(),
            "timestamp": d["timestamp"].to_numpy(np.float64) / 1000.0,
        }
    )
    del d
    # finalize() re-sorts by user and keeps this (timestamp) order within each student.
    df = schema.drop_short_students(df, MIN_SEQ_LEN)
    df["skills"], vocab = schema.densify_skills(df["skills"])
    df = schema.finalize(df)
    meta = schema.DatasetMeta(
        name=NAME,
        n_users=int(df["user_id"].nunique()),
        n_items=int(df["item_id"].nunique()),
        n_skills=len(vocab),
        n_interactions=len(df),
        has_timestamps=True,
    )
    out = data_dir / "processed" / NAME
    schema.save(df, meta, out)
    print(
        f"{NAME}: {meta.n_users:,} students, {meta.n_interactions:,} interactions, "
        f"{meta.n_items:,} questions, {meta.n_skills} skills -> {out}"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--sample", type=int, default=50_000, help="number of students; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    build(args.data_dir, args.sample, args.seed, args.workers)


if __name__ == "__main__":
    main()
