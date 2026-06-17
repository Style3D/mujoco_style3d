import numpy as np

def report_deprecated(func):
    print(f' !!!: "{func.__name__}" is deprecated. will be removed in a future release. !!!')


def write_obj(pos, faces, obj_path: str) -> None:
    with open(obj_path, 'w') as f:
        for v in pos:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')



def read_uv_from_obj(obj_file):
    uv = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line[:3] == 'vt ':
                values = line.split()
                uv.append((float(values[1]), float(values[2])))
    return np.asarray(uv, dtype=float)