"""Download the data of the documentation notebooks from Zenodo into ``docs/data``.

usage: python scripts/fetch_docs_data.py [--force]

The notebooks that read local files (nodes and paths, districts and landmarks from local files,
network cleaning) find their data in ``docs/data``. This script downloads the cityImage
documentation dataset, checks it against its SHA-256 and extracts it there. It does nothing when
the files are already present, unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# The archive of version 24322208 of the dataset (all versions: doi:10.5281/zenodo.24322207)
DATA_URL = "https://zenodo.org/records/24322208/files/cityImage-docs-data.zip?download=1"
DATA_SHA256 = "c3b8332f853c0ad8cbcbad2d3a8db8eab1268c8aa9804801b1b2fe5c18198a27"

DATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "data"
FILES = (
    "Boston/Boston_DTM.tif",
    "Boston/Boston_obstructions.gpkg",
    "Boston/Boston_streetNetwork4000.gpkg",
    "Boston/Boston_studyArea.gpkg",
    "Boston/otherSources/Boston_historic_landmarks.gpkg",
    "Tripoli/Tripoli_edges.gpkg",
    "Tripoli/Tripoli_nodes.gpkg",
    "Tripoli/Tripoli_studyArea.gpkg",
    "Tadoussac/Tadoussac_street_network.gpkg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, target: Path) -> None:
    """Extract the archive into ``target``, refusing members that would land outside it."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"archive member outside the data folder: {member.filename}")
        zf.extractall(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if the files exist")
    args = parser.parse_args(argv)

    missing = [name for name in FILES if not (DATA_DIR / name).exists()]
    if not missing and not args.force:
        print(f"The documentation data are already in {DATA_DIR}.")
        return 0
    if not DATA_URL:
        print("The documentation dataset is not published on Zenodo yet.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "cityImage-docs-data.zip"
        print(f"Downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file)
        digest = _sha256(archive)
        if digest != DATA_SHA256:
            print(f"SHA-256 mismatch: expected {DATA_SHA256}, got {digest}.", file=sys.stderr)
            return 1
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _extract(archive, DATA_DIR)

    missing = [name for name in FILES if not (DATA_DIR / name).exists()]
    if missing:
        print(f"Missing after extraction: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"Documentation data extracted into {DATA_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
