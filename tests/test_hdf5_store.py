from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from backend.hdf5_store import DatasetStore


def write_fixture(path: Path, offset: float = 0.0) -> None:
    rows, cols = 24, 32
    lat = np.linspace(35.0, 34.0, rows)
    lon = np.linspace(-118.0, -117.0, cols)
    yy, xx = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
    unwrapped = offset + yy * 10.0 + xx * 2.0
    wrapped = np.angle(np.exp(1j * unwrapped))
    coherence = np.clip(0.2 + xx * 0.7, 0, 1)
    topo = 1000.0 + yy * 200.0 - xx * 80.0
    data = np.stack([wrapped, unwrapped, coherence, topo]).astype(np.float32)
    data[:, 0, 0] = np.nan

    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=data)
        h5.create_dataset("band_names", data=np.asarray(["wrapped_phase", "unwrapped_phase", "coherence", "topography"], dtype="S"))
        h5.create_dataset("units", data=np.asarray(["rad", "rad", "1", "m"], dtype="S"))
        h5.create_dataset("lat", data=lat)
        h5.create_dataset("lon", data=lon)
        h5.attrs["title"] = f"Synthetic {offset:g}"
        h5.attrs["pair"] = "20200101_20200201"


def test_dataset_discovery_and_summary(tmp_path: Path) -> None:
    write_fixture(tmp_path / "igram_a.h5")
    store = DatasetStore(tmp_path)

    summaries = store.list_datasets()

    assert len(summaries) == 1
    assert summaries[0].id == "igram_a"
    assert summaries[0].shape.rows == 24
    assert summaries[0].bounds.north == 35.0
    assert summaries[0].bounds.west == -118.0
    assert [band.name for band in summaries[0].bands] == ["wrapped_phase", "unwrapped_phase", "coherence", "topography"]


def test_preview_png_is_generated(tmp_path: Path) -> None:
    write_fixture(tmp_path / "igram_a.h5")
    store = DatasetStore(tmp_path)

    png = store.preview_png("igram_a", "unwrapped_phase", cmap="viridis", max_size=64)

    assert png.startswith(b"\x89PNG")
    assert len(png) > 100


def test_sample_point_and_out_of_bounds(tmp_path: Path) -> None:
    write_fixture(tmp_path / "igram_a.h5")
    store = DatasetStore(tmp_path)

    sample = store.sample_point("igram_a", lat=34.5, lon=-117.5, active_band="unwrapped_phase")
    outside = store.sample_point("igram_a", lat=40.0, lon=-117.5, active_band="unwrapped_phase")

    assert sample["in_bounds"] is True
    assert sample["active_band"] == "unwrapped_phase"
    assert sample["active_value"] is not None
    assert sample["values"]["coherence"] is not None
    assert outside["in_bounds"] is False


def test_transect_samples_multiple_datasets(tmp_path: Path) -> None:
    write_fixture(tmp_path / "igram_a.h5", offset=0.0)
    write_fixture(tmp_path / "igram_b.h5", offset=5.0)
    store = DatasetStore(tmp_path)

    result = store.transect(
        ["igram_a", "igram_b"],
        "unwrapped_phase",
        [(34.95, -117.95), (34.05, -117.05)],
        samples=40,
    )

    assert result["band"] == "unwrapped_phase"
    assert len(result["distance_km"]) == 40
    assert len(result["profiles"]) == 2
    assert len(result["profiles"][0]["values"]) == 40
    assert result["profiles"][1]["values"][20] > result["profiles"][0]["values"][20]
