"""
Portable prediction-record container and serializer helpers.

The module is intentionally self-contained so it can be copied between
codebases. It uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


FORMAT_NAME = "prediction_file"
FORMAT_VERSION = 1
GZIP_MAGIC = b"\x1f\x8b"


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
    payload = json.dumps(obj.to_dict(), indent=indent, ensure_ascii=True).encode("utf-8")

    if _should_gzip(path, compress):
        with gzip.open(path, "wb") as f:
            f.write(payload)
    else:
        with open(path, "wb") as f:
            f.write(payload)


def load(path: str | Path) -> PredictionFile:
    path = Path(path)
    return _loads_from_bytes(_read_bytes(path))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
