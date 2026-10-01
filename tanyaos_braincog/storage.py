# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon. Adapted from TanyaOS on 2026-10-01.
"""Explicit isolated storage; no dependency on the TanyaOS protected core."""
from pathlib import Path

def storage_root(base_dir: Path) -> Path:
    directory = Path(base_dir).expanduser().resolve() / "storage"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
