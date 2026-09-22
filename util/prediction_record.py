"""
Portable prediction-record container and serializer helpers.

The module is intentionally self-contained so it can be copied between
codebases. JSON uses only the Python standard library; HDF5 support is enabled
when the optional h5py dependency is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import gzip
from itertools import count
import json
from pathlib import Path
from typing import Any, Iterable


FORMAT_NAME = "prediction_file"
FORMAT_VERSION = 1
GZIP_MAGIC = b"\x1f\x8b"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
HDF5_STORAGE_VERSION = 1
HDF5_DATASET_REF = "__prediction_record_hdf5_dataset__"
HDF5_MIN_ARRAY_BYTES = 4096


def _unwrap_method(value: Any, method_name: str) -> Any:
    method = getattr(value, method_name, None)
    if callable(method):
        return method()
    return value


def to_jsonable(value: Any) -> Any:
    """
    Recursively convert common Python / tensor / ndarray-like objects into a
    JSON-serializable structure.
    """
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Torch-like objects
    value = _unwrap_method(value, "detach")
    value = _unwrap_method(value, "cpu")

    # Numpy scalar-like objects
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        else:
            if isinstance(scalar, (str, int, float, bool)) or scalar is None:
                return scalar
            value = scalar

    # Tensor / ndarray-like objects
    value = _unwrap_method(value, "tolist")

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


@dataclass
class PredictionRecord:
    raw_data: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "raw_data": to_jsonable(self.raw_data),
            "losses": to_jsonable(self.losses),
            "prediction": to_jsonable(self.prediction),
            "meta": to_jsonable(self.meta),
        }
        if self.record_id is not None:
            data["id"] = self.record_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        return cls(
            raw_data=data.get("raw_data", {}),
            losses=data.get("losses", {}),
            prediction=data.get("prediction", {}),
            meta=data.get("meta", {}),
            record_id=data.get("id"),
        )


@dataclass
class PredictionFile:
    dataset_name: str
    codebase_name: str
    records: list[PredictionRecord]
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        summary = dict(self.summary)
        summary.setdefault("num_records", len(self.records))
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "dataset_name": self.dataset_name,
            "codebase_name": self.codebase_name,
            "summary": to_jsonable(summary),
            "records": [record.to_dict() for record in self.records],
            "meta": to_jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionFile":
        records = [PredictionRecord.from_dict(record) for record in data.get("records", [])]
        return cls(
            format_name=data.get("format_name", FORMAT_NAME),
            format_version=data.get("format_version", FORMAT_VERSION),
            dataset_name=data.get("dataset_name", ""),
            codebase_name=data.get("codebase_name", ""),
            summary=data.get("summary", {}),
            records=records,
            meta=data.get("meta", {}),
        )


def make_prediction_file(
    dataset_name: str,
    codebase_name: str,
    records: Iterable[PredictionRecord],
    meta: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> PredictionFile:
    return PredictionFile(
        dataset_name=dataset_name,
        codebase_name=codebase_name,
        records=list(records),
        meta={} if meta is None else meta,
        summary={} if summary is None else summary,
    )


def _should_gzip(path: str | Path, compress: bool | None) -> bool:
    if compress is not None:
        return compress
    path_str = str(path).lower()
    return path_str.endswith(".gz") or path_str.endswith(".json.gz") or path_str.endswith(".gzjson")


def _should_use_hdf5(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".h5", ".hdf5"}


def _require_hdf5():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "HDF5 prediction files require the optional 'h5py' dependency. "
            "Install it with 'pip install h5py'."
        ) from exc
    return h5py, np


def _raw_prediction_file_dict(obj: PredictionFile) -> dict[str, Any]:
    summary = dict(obj.summary)
    summary.setdefault("num_records", len(obj.records))
    records = []
    for record in obj.records:
        data = {
            "raw_data": record.raw_data,
            "losses": record.losses,
            "prediction": record.prediction,
            "meta": record.meta,
        }
        if record.record_id is not None:
            data["id"] = record.record_id
        records.append(data)
    return {
        "format_name": obj.format_name,
        "format_version": obj.format_version,
        "dataset_name": obj.dataset_name,
        "codebase_name": obj.codebase_name,
        "summary": summary,
        "records": records,
        "meta": obj.meta,
    }


def _numeric_array(value: Any, np):
    value = _unwrap_method(value, "detach")
    value = _unwrap_method(value, "cpu")

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        try:
            value = numpy_method()
        except Exception:
            pass

    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)) and value and isinstance(
        value[0], (list, tuple, int, float, bool, complex)
    ):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
    else:
        return None

    if array.dtype.kind not in "biufc":
        return None
    if array.nbytes < HDF5_MIN_ARRAY_BYTES:
        return None
    return array


def _encode_hdf5_value(value: Any, arrays, counter, np, compress: bool):
    if is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}

    array = _numeric_array(value, np)
    if array is not None:
        dataset_name = f"{next(counter):08d}"
        dataset_options = {}
        if compress and array.ndim > 0 and array.size > 0:
            dataset_options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        arrays.create_dataset(dataset_name, data=array, **dataset_options)
        return {HDF5_DATASET_REF: dataset_name}

    if isinstance(value, dict):
        return {
            str(key): _encode_hdf5_value(item, arrays, counter, np, compress)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_encode_hdf5_value(item, arrays, counter, np, compress) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return to_jsonable(value)


def _decode_hdf5_value(value: Any, arrays):
    if isinstance(value, dict):
        if set(value) == {HDF5_DATASET_REF}:
            array = arrays[value[HDF5_DATASET_REF]][()]
            return array.item() if getattr(array, "ndim", 0) == 0 else array.tolist()
        return {key: _decode_hdf5_value(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_hdf5_value(item, arrays) for item in value]
    return value


def _save_hdf5(
    obj: PredictionFile,
    path: Path,
    compress: bool | None,
    indent: int | None,
) -> None:
    h5py, np = _require_hdf5()
    use_compression = compress is not False

    def write_manifest(group, name, value):
        payload = json.dumps(value, indent=indent, ensure_ascii=True).encode("utf-8")
        options = {}
        if use_compression and payload:
            options = {"compression": "gzip", "compression_opts": 4}
        group.create_dataset(
            name,
            data=np.frombuffer(payload, dtype=np.uint8),
            **options,
        )

    with h5py.File(path, "w") as file:
        file.attrs["format_name"] = obj.format_name
        file.attrs["format_version"] = obj.format_version
        file.attrs["hdf5_storage_version"] = HDF5_STORAGE_VERSION
        raw_file = _raw_prediction_file_dict(obj)
        raw_records = raw_file.pop("records")

        header_arrays = file.create_group("arrays")
        header = _encode_hdf5_value(
            raw_file, header_arrays, count(), np, use_compression
        )
        write_manifest(file, "header_json", header)

        records = file.create_group("records")
        for index, raw_record in enumerate(raw_records):
            record_group = records.create_group(f"{index:08d}")
            record_arrays = record_group.create_group("arrays")
            record = _encode_hdf5_value(
                raw_record,
                record_arrays,
                count(),
                np,
                use_compression,
            )
            write_manifest(record_group, "manifest_json", record)


def _load_hdf5(path: Path) -> PredictionFile:
    h5py, _ = _require_hdf5()
    with h5py.File(path, "r") as file:
        storage_version = int(file.attrs.get("hdf5_storage_version", 0))
        if storage_version != HDF5_STORAGE_VERSION:
            raise ValueError(
                f"Unsupported prediction-record HDF5 storage version {storage_version}."
            )
        header = json.loads(file["header_json"][()].tobytes())
        data = _decode_hdf5_value(header, file["arrays"])
        data["records"] = []
        for record_name in sorted(file["records"]):
            record_group = file["records"][record_name]
            record = json.loads(record_group["manifest_json"][()].tobytes())
            data["records"].append(
                _decode_hdf5_value(record, record_group["arrays"])
            )
    return PredictionFile.from_dict(data)


def _read_bytes(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _loads_from_bytes(payload: bytes) -> PredictionFile:
    if payload.startswith(GZIP_MAGIC):
        payload = gzip.decompress(payload)
    data = json.loads(payload.decode("utf-8"))
    return PredictionFile.from_dict(data)


def save(obj: PredictionFile, path: str | Path, compress: bool | None = None, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _should_use_hdf5(path):
        _save_hdf5(obj, path, compress=compress, indent=indent)
        return

    payload = json.dumps(obj.to_dict(), indent=indent, ensure_ascii=True).encode("utf-8")

    if _should_gzip(path, compress):
        with gzip.open(path, "wb") as f:
            f.write(payload)
    else:
        with open(path, "wb") as f:
            f.write(payload)


def load(path: str | Path) -> PredictionFile:
    path = Path(path)
    with open(path, "rb") as file:
        magic = file.read(len(HDF5_MAGIC))
    if magic == HDF5_MAGIC:
        return _load_hdf5(path)
    return _loads_from_bytes(_read_bytes(path))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
