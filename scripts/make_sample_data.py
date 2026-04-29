from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BANDS = ["wrapped_phase", "unwrapped_phase", "coherence", "topography"]
UNITS = ["rad", "rad", "1", "m"]


def write_sample(path: Path, title: str, phase_offset: float) -> None:
    rows, cols = 420, 560
    lat = np.linspace(35.25, 34.65, rows)
    lon = np.linspace(-118.35, -117.45, cols)
    y, x = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")

    bowl = np.exp(-((x * 1.8) ** 2 + (y * 2.2) ** 2))
    ramp = x * 2.0 - y * 1.2
    unwrapped = phase_offset + ramp + bowl * 7.5
    wrapped = np.angle(np.exp(1j * unwrapped))
    coherence = np.clip(0.85 - np.abs(y) * 0.28 - np.abs(x) * 0.18 + bowl * 0.08, 0.05, 1.0)
    topography = 1200 + y * 450 - x * 160 + np.sin(x * 7) * 45

    mask = ((x + 0.45) ** 2 + (y - 0.35) ** 2) < 0.045
    stack = np.stack([wrapped, unwrapped, coherence, topography]).astype(np.float32)
    stack[:, mask] = np.nan

    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=stack, compression="gzip", compression_opts=4)
        h5.create_dataset("band_names", data=np.asarray(BANDS, dtype="S"))
        h5.create_dataset("units", data=np.asarray(UNITS, dtype="S"))
        h5.create_dataset("lat", data=lat)
        h5.create_dataset("lon", data=lon)
        h5.attrs["title"] = title
        h5.attrs["pair"] = "20200101_20200201"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    write_sample(DATA_DIR / "sample_interferogram_a.h5", "Sample Interferogram A", 0.0)
    write_sample(DATA_DIR / "sample_interferogram_b.h5", "Sample Interferogram B", 1.7)
    print(f"Wrote sample HDF5 files to {DATA_DIR}")


if __name__ == "__main__":
    main()
