"""Independent accelerometer/gyroscope provenance readers for G10."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawIMURecord:
    timestamps_s: np.ndarray
    acceleration_mps2: np.ndarray
    gyroscope_rads: np.ndarray
    sensor_location: str
    provenance: str
    source_path: Path


def _validate(timestamps_s: np.ndarray, acceleration: np.ndarray, gyroscope: np.ndarray, path: Path) -> RawIMURecord:
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    gyroscope = np.asarray(gyroscope, dtype=np.float64)
    if timestamps_s.ndim != 1 or acceleration.shape != (len(timestamps_s), 3) or gyroscope.shape != (len(timestamps_s), 3):
        raise ValueError(f"Invalid IMU shapes in {path}: ts={timestamps_s.shape}, acc={acceleration.shape}, gyro={gyroscope.shape}")
    if len(timestamps_s) < 2 or not np.isfinite(timestamps_s).all() or not np.all(np.diff(timestamps_s) > 0):
        raise ValueError(f"Timestamps must be finite and strictly increasing in {path}")
    if not np.isfinite(acceleration).all() or not np.isfinite(gyroscope).all():
        raise ValueError(f"Non-finite acceleration/gyro in {path}")
    return RawIMURecord(timestamps_s, acceleration, gyroscope, "", "", path)


def parse_totalcapture_auxfields(path: str | Path, sensor_name: str = "L_LowArm") -> RawIMURecord:
    """Parse Xsens AuxFields acceleration and gyro for one sensor.

    AuxFields stores acceleration in dataset-native acceleration units and
    angular velocity in rad/s.  The existing project protocol treats these as
    SI values; no resampling or filtering is performed here.
    """
    source = Path(path).expanduser().resolve()
    lines = iter(source.read_text(encoding="utf-8").splitlines())
    header = next(lines).split()
    n_sensors, n_frames = int(header[0]), int(header[1])
    acceleration: list[list[float]] = []
    gyroscope: list[list[float]] = []
    for _ in range(n_frames):
        next(lines)
        found = False
        for _ in range(n_sensors):
            parts = next(lines).split()
            if parts[0] != sensor_name:
                continue
            if len(parts) < 14:
                raise ValueError(f"Expected AuxFields 14-column row in {source}: {parts}")
            acceleration.append([float(value) for value in parts[5:8]])
            gyroscope.append([float(value) for value in parts[8:11]])
            found = True
        if not found:
            raise ValueError(f"Missing {sensor_name} in frame of {source}")
    record = _validate(np.arange(n_frames, dtype=np.float64) / 60.0, acceleration, gyroscope, source)
    return RawIMURecord(record.timestamps_s, record.acceleration_mps2, record.gyroscope_rads, sensor_name, "totalcapture_xsens_auxfields_measured", source)


def parse_egohumans_realistic(path: str | Path, sensor_name: str = "LeftWrist") -> RawIMURecord:
    """Read realistic-IMU acceleration/gyro generated from SMPL kinematics."""
    source = Path(path).expanduser().resolve()
    payload = np.load(source, allow_pickle=True).item()
    metadata = payload.get("metadata", {})
    names = [str(name) for name in metadata.get("sensor_names", [])]
    if sensor_name not in names:
        raise ValueError(f"Sensor {sensor_name!r} not found in {source}; available={names}")
    idx = names.index(sensor_name)
    fps = float(metadata.get("target_fps", 0.0))
    if fps <= 0:
        raise ValueError(f"Missing positive metadata.target_fps in {source}")
    acceleration = np.asarray(payload["acc"][:, idx], dtype=np.float64)
    gyroscope = np.asarray(payload["gyro"][:, idx], dtype=np.float64)
    record = _validate(np.arange(len(acceleration), dtype=np.float64) / fps, acceleration, gyroscope, source)
    return RawIMURecord(record.timestamps_s, record.acceleration_mps2, record.gyroscope_rads, sensor_name, "egohumans_realistic_smpl_kinematics", source)


def _column(row: dict[str, str], name: str) -> str:
    if name in row:
        return name
    raise ValueError(f"Missing CSV column {name!r}; available={list(row)}")


def parse_custom_csv(path: str | Path, sensor_name: str = "left_wrist") -> RawIMURecord:
    """Parse Custom CSV, converting g→m/s² and degree/s→rad/s."""
    source = Path(path).expanduser().resolve()
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty IMU CSV {source}")
    names = {key: _column(rows[0], key) for key in ("epoch_ms", "加速度X(g)", "加速度Y(g)", "加速度Z(g)", "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)")}
    timestamps = np.asarray([float(row[names["epoch_ms"]]) for row in rows], dtype=np.float64) / 1000.0
    timestamps -= timestamps[0]
    acceleration = np.asarray([[float(row[names[f"加速度{axis}(g)"]]) for axis in "XYZ"] for row in rows], dtype=np.float64) * 9.80665
    gyroscope = np.asarray([[float(row[names[f"角速度{axis}(°/s)"]]) for axis in "XYZ"] for row in rows], dtype=np.float64) * (np.pi / 180.0)
    record = _validate(timestamps, acceleration, gyroscope, source)
    return RawIMURecord(record.timestamps_s, record.acceleration_mps2, record.gyroscope_rads, sensor_name, "custom_device_measured_csv", source)


def parse_custom_csv_series(paths: list[str | Path], sensor_name: str = "left_wrist") -> RawIMURecord:
    """Parse multiple segmented Custom CSVs on one absolute epoch timeline.

    Segment exports restart their local row index but retain ``epoch_ms``.  This
    reader therefore keeps the epoch values, sorts all rows, and removes exact
    duplicate timestamps before returning seconds relative to the first sample.
    """
    sources = [Path(path).expanduser().resolve() for path in paths]
    if not sources:
        raise ValueError("Custom CSV series requires at least one path")
    timestamps_ms: list[float] = []
    acceleration: list[list[float]] = []
    gyroscope: list[list[float]] = []
    for source in sources:
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        names = {key: _column(rows[0], key) for key in ("epoch_ms", "加速度X(g)", "加速度Y(g)", "加速度Z(g)", "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)")}
        for row in rows:
            timestamps_ms.append(float(row[names["epoch_ms"]]))
            acceleration.append([float(row[names[f"加速度{axis}(g)"]]) * 9.80665 for axis in "XYZ"])
            gyroscope.append([float(row[names[f"角速度{axis}(°/s)"]]) * (np.pi / 180.0) for axis in "XYZ"])
    if len(timestamps_ms) < 2:
        raise ValueError(f"Custom CSV series has fewer than two samples: {sources}")
    order = np.argsort(np.asarray(timestamps_ms, dtype=np.float64), kind="stable")
    ts = np.asarray(timestamps_ms, dtype=np.float64)[order]
    acc = np.asarray(acceleration, dtype=np.float64)[order]
    gyro = np.asarray(gyroscope, dtype=np.float64)[order]
    keep = np.r_[True, np.diff(ts) > 0]
    record = _validate((ts[keep] - ts[keep][0]) / 1000.0, acc[keep], gyro[keep], sources[0])
    return RawIMURecord(record.timestamps_s, record.acceleration_mps2, record.gyroscope_rads, sensor_name, "custom_device_measured_csv_segment_series", sources[0])
