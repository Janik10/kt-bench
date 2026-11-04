"""Dataset loading. ``load(name)`` builds the processed Parquet on first use."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ktbench.data import schema


def load(name: str, data_dir: Path = Path("data")) -> tuple[pd.DataFrame, schema.DatasetMeta, Path]:
    out = data_dir / "processed" / name
    if not (out / "interactions.parquet").exists():
        if name == "assist2009":
            from ktbench.data import assist2009

            assist2009.build(data_dir)
        elif name == "ednet":
            raise FileNotFoundError("run `python -m ktbench.data.ednet` first (see README)")
        elif name.startswith("synthetic"):
            from ktbench.data import synthetic

            df, meta = synthetic.generate()
            schema.save(df, meta, out)
        else:
            raise ValueError(f"unknown dataset {name!r}")
    df, meta = schema.load(out)
    return df, meta, out
