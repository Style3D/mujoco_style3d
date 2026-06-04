import numpy as np

_VTK_SECTION_HEADERS = {
    'POINTS',
    'CELLS',
    'OFFSETS',
    'CONNECTIVITY',
    'CELL_TYPES',
    'POINT_DATA',
    'CELL_DATA',
    'FIELD',
}


def _next_nonempty_line(lines, i):
    while i < len(lines) and not lines[i].split():
        i += 1
    return i


def _read_numbers_until_section(lines, i, cast):
    values = []
    while i < len(lines):
        parts = lines[i].split()
        if parts and parts[0] in _VTK_SECTION_HEADERS:
            break
        for part in parts:
            values.append(cast(part))
        i += 1
    return values, i


# load tetrahedron mesh from a VTK unstructured grid file (ASCII)
def load_tetrahedrons(file_path):
    vertices = []
    tets = []

    with open(file_path, 'r') as f:
        lines = [l.rstrip() for l in f]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue

        if parts[0] == 'POINTS':
            n_points = int(parts[1])
            i += 1
            coords = []
            while len(coords) < n_points * 3:
                row = lines[i].split()
                i += 1
                if not row or row[0].startswith('#'):
                    continue
                coords.extend(float(v) for v in row)
            vertices = np.array(coords[:n_points * 3], dtype=float).reshape(-1, 3).tolist()

        elif parts[0] == 'CELLS':
            n_cells = int(parts[1])
            i += 1
            i = _next_nonempty_line(lines, i)

            # VTK 5 legacy ASCII uses:
            #   CELLS <num_offsets> <num_connectivity>
            #   OFFSETS <type>
            #   ...
            #   CONNECTIVITY <type>
            #   ...
            if i < len(lines) and lines[i].split()[0] == 'OFFSETS':
                i += 1
                offsets, i = _read_numbers_until_section(lines, i, int)

                i = _next_nonempty_line(lines, i)
                if i >= len(lines) or lines[i].split()[0] != 'CONNECTIVITY':
                    raise ValueError(f'VTK CELLS block in {file_path} has OFFSETS but no CONNECTIVITY')

                i += 1
                connectivity, i = _read_numbers_until_section(lines, i, int)

                for start, end in zip(offsets[:-1], offsets[1:]):
                    if end - start == 4:
                        tets.append(connectivity[start:end])

            # VTK 3 legacy ASCII uses one cell per row:
            #   4 i j k l
            else:
                for _ in range(n_cells):
                    row = lines[i].split()
                    n_verts = int(row[0])
                    if n_verts == 4:
                        tets.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
                    i += 1

        else:
            i += 1

    return np.array(vertices, dtype=float), np.array(tets, dtype=int)

def compute_boundary_faces(tets: np.ndarray):
    """Returns compacted boundary triangle faces and their source vertex indices."""
    # Each tet has 4 triangular faces defined by combinations of its 4 vertices
    face_local = np.array([[1,0,2],[0,1,3],[0,3,2],[1,2,3]])
    # Build all faces: shape (N*4, 3)
    all_faces = tets[:, face_local].reshape(-1, 3)
    # Canonical form: sort each face's vertex indices so shared faces compare equal
    sorted_faces = np.sort(all_faces, axis=1)
    # Find faces that appear exactly once (boundary = not shared)
    _, inverse, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    boundary_faces = all_faces[boundary_mask]
    used_vertices, compact_faces = np.unique(boundary_faces, return_inverse=True)
    return compact_faces.reshape(-1, 3), used_vertices
