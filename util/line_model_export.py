"""Export 3D line segments as native GLB line primitives."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942


def _as_lines(lines: Any) -> np.ndarray:
    detach = getattr(lines, 'detach', None)
    if callable(detach):
        lines = detach()
    cpu = getattr(lines, 'cpu', None)
    if callable(cpu):
        lines = cpu()
    numpy = getattr(lines, 'numpy', None)
    if callable(numpy):
        lines = numpy()
    array = np.asarray(lines, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2, 3), dtype=np.float32)
    return array.reshape(-1, 2, 3)


def _line_vertices(lines: np.ndarray) -> np.ndarray:
    finite = np.isfinite(lines).all(axis=(1, 2))
    nondegenerate = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1) > 1e-12
    return np.asarray(lines[finite & nondegenerate], dtype='<f4').reshape(-1, 3)


def _pad(data: bytes, padding: bytes) -> bytes:
    return data + padding * ((-len(data)) % 4)


def save_line_model_glb(
    path: str | Path,
    predicted_lines: Any,
    ground_truth_lines: Any | None = None,
    *,
    prediction_name: str = 'predictions',
) -> Path:
    """Save predicted and optional ground-truth segments as GLB edges."""
    path = Path(path)
    predicted = _as_lines(predicted_lines)
    ground_truth = _as_lines(ground_truth_lines) if ground_truth_lines is not None else _as_lines([])

    groups = [
        (prediction_name, predicted, [1.0, 0.28, 0.05, 1.0]),
    ]
    if ground_truth_lines is not None:
        groups.append(('ground_truth', ground_truth, [0.05, 0.75, 1.0, 1.0]))

    document = {
        'asset': {'version': '2.0', 'generator': 'LINEA line_model_export'},
        'scene': 0,
        'scenes': [{'nodes': []}],
        'nodes': [],
        'meshes': [],
        'materials': [],
        'accessors': [],
        'bufferViews': [],
        'buffers': [{'byteLength': 0}],
        'extensionsUsed': ['KHR_materials_unlit'],
        'extras': {'coordinate_system': 'camera'},
    }
    binary = bytearray()

    for name, lines, color in groups:
        vertices = _line_vertices(lines)
        if len(vertices) == 0:
            continue

        vertex_offset = len(binary)
        vertex_bytes = vertices.tobytes()
        binary.extend(vertex_bytes)
        while len(binary) % 4:
            binary.append(0)

        vertex_view = len(document['bufferViews'])
        document['bufferViews'].append({
            'buffer': 0,
            'byteOffset': vertex_offset,
            'byteLength': len(vertex_bytes),
            'target': 34962,
        })

        position_accessor = len(document['accessors'])
        document['accessors'].append({
            'bufferView': vertex_view,
            'componentType': 5126,
            'count': len(vertices),
            'type': 'VEC3',
            'min': vertices.min(axis=0).tolist(),
            'max': vertices.max(axis=0).tolist(),
        })
        material = len(document['materials'])
        document['materials'].append({
            'name': name,
            'pbrMetallicRoughness': {
                'baseColorFactor': color,
                'metallicFactor': 0.0,
                'roughnessFactor': 0.8,
            },
            'doubleSided': True,
            'extensions': {'KHR_materials_unlit': {}},
        })
        mesh = len(document['meshes'])
        document['meshes'].append({
            'name': name,
            'primitives': [{
                'attributes': {'POSITION': position_accessor},
                'material': material,
                'mode': 1,
            }],
            'extras': {'line_count': len(vertices) // 2},
        })
        node = len(document['nodes'])
        document['nodes'].append({'name': name, 'mesh': mesh})
        document['scenes'][0]['nodes'].append(node)

    document['buffers'][0]['byteLength'] = len(binary)
    json_chunk = _pad(json.dumps(document, separators=(',', ':')).encode('utf-8'), b' ')
    bin_chunk = _pad(bytes(binary), b'\x00')
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = bytearray(struct.pack('<III', GLB_MAGIC, GLB_VERSION, total_length))
    glb.extend(struct.pack('<II', len(json_chunk), JSON_CHUNK_TYPE))
    glb.extend(json_chunk)
    glb.extend(struct.pack('<II', len(bin_chunk), BIN_CHUNK_TYPE))
    glb.extend(bin_chunk)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return path
