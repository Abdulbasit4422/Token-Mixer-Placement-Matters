from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from omegaconf import DictConfig

from token_mixer.data.prepare import prepare_brats


def run_prepare(cfg: DictConfig) -> None:
    """Prepare configured BraTS data and write a compact case index."""
    source_root = Path(cfg.paths.source_root)
    data_root = Path(cfg.paths.data_root).resolve()
    cases = prepare_brats(source_root, data_root)

    index = [
        {
            "case_id": case.case_id,
            "modalities": {
                name: path.relative_to(data_root).as_posix()
                for name, path in case.modalities.items()
            },
            "segmentation": case.segmentation.relative_to(data_root).as_posix(),
        }
        for case in cases
    ]
    index_path = data_root / "case_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_index_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.",
        suffix=".tmp",
        dir=str(index_path.parent),
    )
    temporary_index = Path(temporary_index_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_index, index_path)
    finally:
        if temporary_index.exists():
            temporary_index.unlink()

    print(f"Prepared {len(cases)} cases.")
    print(f"Case index: {index_path}")
