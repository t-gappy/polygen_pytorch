import numpy as np
from .preprocess import redirect_same_vertices, reorder_vertices, reorder_faces, bit_quantization


def read_objfile(file_path):
    vertices = []
    normals = []
    faces = []
    
    with open(file_path) as fr:
        for line in fr:
            data = line.split()
            if len(data) > 0:
                if data[0] == "v":
                    vertices.append(data[1:])
                elif data[0] == "vn":
                    normals.append(data[1:])
                elif data[0] == "f":
                    face = np.array([
                        [int(p.split("/")[0]), int(p.split("/")[2])]
                        for p in data[1:]
                    ]) - 1
                    faces.append(face)
    
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    return vertices, normals, faces

    
def load_pipeline(file_path, bit=8, remove_normal_ids=True):
    vs, ns, fs = read_objfile(file_path)
    
    vs = bit_quantization(vs, bit=bit)
    vs, fs = redirect_same_vertices(vs, fs)
    
    vs, ids = reorder_vertices(vs)
    fs = reorder_faces(fs, ids)
    
    if remove_normal_ids:
        fs = [f[:, 0] for f in fs]
        
    return vs, ns, fs