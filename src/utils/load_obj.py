import numpy as np
import pandas as pd
import open3d as o3d
from .preprocess import redirect_same_vertices, reorder_vertices, reorder_faces, bit_quantization


def read_objfile(file_path, return_o3d=False):
    
    obj = o3d.io.read_triangle_mesh(file_path)
    if return_o3d:
        return obj
    else:
        v = np.asarray(obj.vertices, dtype=np.float32)
        f = np.asarray(obj.triangles, dtype=np.int32)
        return v, f

    
def load_pipeline(file_path, bit=8):
    v, f = read_objfile(file_path)
    v, f = redirect_same_vertices(v, f)
    
    columns = ["z", "y", "x"]
    v = pd.DataFrame(v, columns=columns)
    v = reorder_vertices(v, columns=columns, add_sorted_index=True)
    
    f = reorder_faces(v, f)
    v = v.values[:, :3]
    
    v = bit_quantization(v, bit=bit)
    f = f.values
    return v, f
