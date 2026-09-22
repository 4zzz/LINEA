"""Export 3D line segments as lightweight GLB tube models."""

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


def _automatic_radius(*line_groups: np.ndarray) -> float:
    points = [lines.reshape(-1, 3) for lines in line_groups if lines.size]
    if not points:
        return 1e-3
    points = np.concatenate(points, axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return 1e-3
    extent = np.ptp(finite, axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal * 0.0015, 1e-6)


def _tube_mesh(lines: np.ndarray, radius: float, sides: int = 8):
    vertices = []
    triangles = []
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)

    for endpoints in lines:
        start, end = endpoints
        direction = end - start
        length = float(np.linalg.norm(direction))
        if not np.isfinite(endpoints).all() or length <= 1e-12:
            continue
        direction /= length
        helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(direction, helper))) > 0.9:
            helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        basis_u = np.cross(direction, helper)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(direction, basis_u)

        first_vertex = len(vertices)
        for endpoint in (start, end):
            for angle in angles:
                offset = radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
                vertices.append(endpoint + offset)

        for side in range(sides):
            next_side = (side + 1) % sides
            a = first_vertex + side
            b = first_vertex + next_side
            c = first_vertex + sides + side
            d = first_vertex + sides + next_side
            triangles.extend(((a, c, b), (b, c, d)))

    return (
        np.asarray(vertices, dtype='<f4').reshape(-1, 3),
        np.asarray(triangles, dtype='<u4').reshape(-1, 3),
    )


def _pad(data: bytes, padding: bytes) -> bytes:
    return data + padding * ((-len(data)) % 4)


def save_line_model_glb(
    path: str | Path,
    predicted_lines: Any,
    ground_truth_lines: Any | None = None,
    *,
    line_radius: float | None = None,
    prediction_name: str = 'predictions',
) -> Path:
    """Save predicted and optional ground-truth lines as colored tube meshes."""
    path = Path(path)
    predicted = _as_lines(predicted_lines)
    ground_truth = _as_lines(ground_truth_lines) if ground_truth_lines is not None else _as_lines([])
    radius = _automatic_radius(predicted, ground_truth) if line_radius is None else line_radius
    if radius <= 0:
        raise ValueError('line_radius must be positive.')

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
        'extras': {'coordinate_system': 'camera', 'line_radius': radius},
    }
    binary = bytearray()

    for name, lines, color in groups:
        vertices, triangles = _tube_mesh(lines, radius)
        if len(vertices) == 0:
            continue

        vertex_offset = len(binary)
        vertex_bytes = vertices.tobytes()
        binary.extend(vertex_bytes)
        while len(binary) % 4:
            binary.append(0)
        index_offset = len(binary)
        index_bytes = triangles.reshape(-1).tobytes()
        binary.extend(index_bytes)
        while len(binary) % 4:
            binary.append(0)

        vertex_view = len(document['bufferViews'])
        document['bufferViews'].append({
            'buffer': 0,
            'byteOffset': vertex_offset,
            'byteLength': len(vertex_bytes),
            'target': 34962,
        })
        index_view = len(document['bufferViews'])
        document['bufferViews'].append({
            'buffer': 0,
            'byteOffset': index_offset,
            'byteLength': len(index_bytes),
            'target': 34963,
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
        index_accessor = len(document['accessors'])
        document['accessors'].append({
            'bufferView': index_view,
            'componentType': 5125,
            'count': triangles.size,
            'type': 'SCALAR',
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
                'indices': index_accessor,
                'material': material,
                'mode': 4,
            }],
            'extras': {'line_count': int(len(lines))},
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
