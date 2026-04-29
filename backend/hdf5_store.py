from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insardash-matplotlib"))
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import map_coordinates

from .schemas import BandInfo, Bounds, DatasetSummary, RasterShape


HDF5_SUFFIXES = {".h5", ".hdf5"}


class DatasetError(ValueError):
    """Raised when an HDF5 file does not match the MVP data contract."""


@dataclass(frozen=True)
class DatasetRef:
    id: str
    path: Path
    mtime_ns: int
    size: int


@dataclass
class DatasetAxes:
    lat: np.ndarray
    lon: np.ndarray

    @cached_property
    def bounds(self) -> Bounds:
        return Bounds(
            south=float(np.nanmin(self.lat)),
            west=float(np.nanmin(self.lon)),
            north=float(np.nanmax(self.lat)),
            east=float(np.nanmax(self.lon)),
        )


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_string_array(values: Any) -> list[str]:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    decoded: list[str] = []
    for item in array.tolist():
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _finite_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _dataset_id(path: Path) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)
    return safe or "dataset"


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


class DatasetStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._summary_cache: dict[tuple[Path, int, int], DatasetSummary] = {}
        self._preview_cache: dict[tuple[Any, ...], bytes] = {}

    def list_refs(self) -> list[DatasetRef]:
        if not self.data_dir.exists():
            return []
        refs: list[DatasetRef] = []
        seen: dict[str, int] = {}
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in HDF5_SUFFIXES:
                continue
            stat = path.stat()
            base_id = _dataset_id(path)
            count = seen.get(base_id, 0)
            seen[base_id] = count + 1
            dataset_id = base_id if count == 0 else f"{base_id}_{count + 1}"
            refs.append(DatasetRef(dataset_id, path, stat.st_mtime_ns, stat.st_size))
        return refs

    def list_datasets(self) -> list[DatasetSummary]:
        return [self.summary(ref.id) for ref in self.list_refs()]

    def summary(self, dataset_id: str) -> DatasetSummary:
        ref = self._get_ref(dataset_id)
        key = (ref.path, ref.mtime_ns, ref.size)
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached.model_copy(deep=True)
        summary = self._read_summary(ref)
        self._summary_cache[key] = summary
        return summary.model_copy(deep=True)

    def preview_png(
        self,
        dataset_id: str,
        band: str,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        max_size: int = 1200,
    ) -> bytes:
        ref = self._get_ref(dataset_id)
        cache_key = (dataset_id, ref.mtime_ns, ref.size, band, cmap, vmin, vmax, max_size)
        cached = self._preview_cache.get(cache_key)
        if cached is not None:
            return cached

        with h5py.File(ref.path, "r") as h5:
            data = self._require_data(h5)
            axes = self._read_axes(h5, data.shape[1], data.shape[2])
            band_index = self._band_index(h5, band, data.shape[0])
            array = np.asarray(data[band_index], dtype=np.float32)

        display = self._orient_for_display(array, axes)
        stride = max(1, math.ceil(max(display.shape) / max_size))
        display = display[::stride, ::stride]
        image = self._colorize(display, cmap, vmin, vmax)

        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        png = buffer.getvalue()
        self._preview_cache[cache_key] = png
        return png

    def sample_point(
        self,
        dataset_id: str,
        lat: float,
        lon: float,
        active_band: str | None = None,
    ) -> dict[str, Any]:
        ref = self._get_ref(dataset_id)
        with h5py.File(ref.path, "r") as h5:
            data = self._require_data(h5)
            axes = self._read_axes(h5, data.shape[1], data.shape[2])
            summary = self.summary(dataset_id)
            row_float, col_float, in_bounds = self._latlon_to_fractional_index(axes, lat, lon)
            if not in_bounds:
                return {
                    "dataset_id": dataset_id,
                    "title": summary.title,
                    "in_bounds": False,
                    "active_band": active_band,
                    "values": {},
                    "units": {band.name: band.units for band in summary.bands},
                }

            row = int(round(row_float))
            col = int(round(col_float))
            row = min(max(row, 0), data.shape[1] - 1)
            col = min(max(col, 0), data.shape[2] - 1)
            raw_values = np.asarray(data[:, row, col], dtype=np.float64)
            values = {
                band.name: _finite_or_none(float(raw_values[band.index]))
                for band in summary.bands
            }
            units = {band.name: band.units for band in summary.bands}
            chosen_band = active_band if active_band in values else None
            return {
                "dataset_id": dataset_id,
                "title": summary.title,
                "in_bounds": True,
                "row": row,
                "col": col,
                "active_band": chosen_band,
                "active_value": values.get(chosen_band) if chosen_band else None,
                "values": values,
                "units": units,
            }

    def transect(
        self,
        dataset_ids: list[str],
        band: str,
        points: list[tuple[float, float]],
        samples: int,
    ) -> dict[str, Any]:
        lats, lons, distance_km = self._interpolate_polyline(points, samples)
        profiles: list[dict[str, Any]] = []
        for dataset_id in dataset_ids:
            ref = self._get_ref(dataset_id)
            with h5py.File(ref.path, "r") as h5:
                data = self._require_data(h5)
                band_index = self._band_index(h5, band, data.shape[0])
                axes = self._read_axes(h5, data.shape[1], data.shape[2])
                rows, cols, in_bounds = self._latlon_arrays_to_indices(axes, lats, lons)
                raster = np.asarray(data[band_index], dtype=np.float32)
                values = map_coordinates(
                    raster,
                    np.vstack([rows, cols]),
                    order=1,
                    mode="constant",
                    cval=np.nan,
                )
                values = np.where(in_bounds, values, np.nan)
                summary = self.summary(dataset_id)
                band_info = next(item for item in summary.bands if item.name == band)
                profiles.append(
                    {
                        "dataset_id": dataset_id,
                        "title": summary.title,
                        "band": band,
                        "units": band_info.units,
                        "values": [_finite_or_none(float(value)) for value in values],
                    }
                )

        return {
            "band": band,
            "distance_km": [float(value) for value in distance_km],
            "lat": [float(value) for value in lats],
            "lon": [float(value) for value in lons],
            "profiles": profiles,
        }

    def _get_ref(self, dataset_id: str) -> DatasetRef:
        for ref in self.list_refs():
            if ref.id == dataset_id:
                return ref
        raise KeyError(f"Unknown dataset id: {dataset_id}")

    def _read_summary(self, ref: DatasetRef) -> DatasetSummary:
        with h5py.File(ref.path, "r") as h5:
            data = self._require_data(h5)
            n_bands, rows, cols = data.shape
            axes = self._read_axes(h5, rows, cols)
            band_names = self._read_band_names(h5, n_bands)
            units = self._read_units(h5, n_bands)
            attrs = {key: _decode_scalar(value) for key, value in h5.attrs.items()}
            title = str(attrs.get("title") or attrs.get("name") or ref.path.stem)
            metadata = {
                key: value
                for key, value in attrs.items()
                if isinstance(value, (str, int, float, bool)) and key not in {"title", "name"}
            }
            return DatasetSummary(
                id=ref.id,
                filename=ref.path.name,
                title=title,
                shape=RasterShape(bands=n_bands, rows=rows, cols=cols),
                bounds=axes.bounds,
                bands=[BandInfo(name=name, index=index, units=units[index]) for index, name in enumerate(band_names)],
                metadata=metadata,
            )

    @staticmethod
    def _require_data(h5: h5py.File) -> h5py.Dataset:
        if "data" not in h5:
            raise DatasetError("HDF5 file must include a /data dataset")
        data = h5["data"]
        if not isinstance(data, h5py.Dataset) or data.ndim != 3:
            raise DatasetError("/data must be a 3D dataset shaped (N_bands, Ny, Nx)")
        return data

    @staticmethod
    def _read_axes(h5: h5py.File, rows: int, cols: int) -> DatasetAxes:
        if "lat" not in h5 or "lon" not in h5:
            raise DatasetError("HDF5 file must include /lat and /lon datasets")
        lat_raw = np.asarray(h5["lat"], dtype=np.float64)
        lon_raw = np.asarray(h5["lon"], dtype=np.float64)

        if lat_raw.ndim == 1 and lon_raw.ndim == 1:
            if lat_raw.size != rows or lon_raw.size != cols:
                raise DatasetError("1D /lat and /lon lengths must match /data rows and columns")
            lat_axis = lat_raw
            lon_axis = lon_raw
        elif lat_raw.shape == (rows, cols) and lon_raw.shape == (rows, cols):
            lat_axis = lat_raw[:, 0]
            lon_axis = lon_raw[0, :]
        else:
            raise DatasetError("/lat and /lon must both be 1D axes or 2D arrays matching /data")

        if not np.all(np.isfinite(lat_axis)) or not np.all(np.isfinite(lon_axis)):
            raise DatasetError("/lat and /lon must contain finite values")
        if lat_axis.size < 2 or lon_axis.size < 2:
            raise DatasetError("/lat and /lon must contain at least two coordinates")
        if not (np.all(np.diff(lat_axis) > 0) or np.all(np.diff(lat_axis) < 0)):
            raise DatasetError("/lat must be monotonic for the MVP regular-grid renderer")
        if not (np.all(np.diff(lon_axis) > 0) or np.all(np.diff(lon_axis) < 0)):
            raise DatasetError("/lon must be monotonic for the MVP regular-grid renderer")
        return DatasetAxes(lat=lat_axis, lon=lon_axis)

    @staticmethod
    def _read_band_names(h5: h5py.File, n_bands: int) -> list[str]:
        if "band_names" in h5:
            names = _decode_string_array(h5["band_names"][()])
        elif "band_names" in h5["data"].attrs:
            names = _decode_string_array(h5["data"].attrs["band_names"])
        else:
            names = [f"band_{index}" for index in range(n_bands)]
        if len(names) != n_bands:
            raise DatasetError("/band_names length must match the first /data dimension")
        if len(set(names)) != len(names):
            raise DatasetError("/band_names entries must be unique")
        return names

    @staticmethod
    def _read_units(h5: h5py.File, n_bands: int) -> list[str | None]:
        if "units" in h5:
            units: list[str | None] = _decode_string_array(h5["units"][()])
        elif "units" in h5["data"].attrs:
            units = _decode_string_array(h5["data"].attrs["units"])
        else:
            units = [None] * n_bands
        if len(units) != n_bands:
            return [None] * n_bands
        return units

    def _band_index(self, h5: h5py.File, band: str, n_bands: int) -> int:
        names = self._read_band_names(h5, n_bands)
        if band not in names:
            raise KeyError(f"Band {band!r} is not available")
        return names.index(band)

    @staticmethod
    def _orient_for_display(array: np.ndarray, axes: DatasetAxes) -> np.ndarray:
        display = array
        if axes.lat[0] < axes.lat[-1]:
            display = np.flipud(display)
        if axes.lon[0] > axes.lon[-1]:
            display = np.fliplr(display)
        return display

    @staticmethod
    def _colorize(array: np.ndarray, cmap_name: str, vmin: float | None, vmax: float | None) -> Image.Image:
        try:
            cmap = colormaps[cmap_name]
        except KeyError as exc:
            raise KeyError(f"Unknown colormap: {cmap_name}") from exc

        finite = np.isfinite(array)
        if vmin is None or vmax is None:
            if finite.any():
                low, high = np.nanpercentile(array[finite], [2, 98])
            else:
                low, high = 0.0, 1.0
            vmin = float(low) if vmin is None else vmin
            vmax = float(high) if vmax is None else vmax
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

        normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = (cmap(normalized) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    @staticmethod
    def _axis_to_fractional_index(axis: np.ndarray, value: float) -> tuple[float, bool]:
        lower = min(float(axis[0]), float(axis[-1]))
        upper = max(float(axis[0]), float(axis[-1]))
        if value < lower or value > upper:
            return math.nan, False
        if axis[0] < axis[-1]:
            return float(np.interp(value, axis, np.arange(axis.size))), True
        return float(np.interp(value, axis[::-1], np.arange(axis.size - 1, -1, -1))), True

    def _latlon_to_fractional_index(self, axes: DatasetAxes, lat: float, lon: float) -> tuple[float, float, bool]:
        row, lat_ok = self._axis_to_fractional_index(axes.lat, lat)
        col, lon_ok = self._axis_to_fractional_index(axes.lon, lon)
        return row, col, bool(lat_ok and lon_ok)

    def _latlon_arrays_to_indices(
        self,
        axes: DatasetAxes,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.empty_like(lats, dtype=np.float64)
        cols = np.empty_like(lons, dtype=np.float64)
        in_bounds = np.ones_like(lats, dtype=bool)
        for index, (lat, lon) in enumerate(zip(lats, lons)):
            row, col, ok = self._latlon_to_fractional_index(axes, float(lat), float(lon))
            rows[index] = row if ok else -1.0
            cols[index] = col if ok else -1.0
            in_bounds[index] = ok
        return rows, cols, in_bounds

    @staticmethod
    def _interpolate_polyline(points: list[tuple[float, float]], samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.asarray(points, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] != 2:
            raise ValueError("A transect requires at least two lat/lon points")

        segment_lengths = _haversine_km(
            coordinates[:-1, 0],
            coordinates[:-1, 1],
            coordinates[1:, 0],
            coordinates[1:, 1],
        )
        total = float(segment_lengths.sum())
        if total <= 0:
            raise ValueError("Transect length must be greater than zero")

        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        target = np.linspace(0.0, total, samples)
        lats = np.interp(target, cumulative, coordinates[:, 0])
        lons = np.interp(target, cumulative, coordinates[:, 1])
        return lats, lons, target
