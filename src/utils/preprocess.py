import numpy as np
import pandas as pd


def redirect_same_vertices(vertices, faces):
    
    faces_with_coord = []
    for face in faces:
        faces_with_coord.append([tuple(vertices[idx]) for idx in face])
    
    coord_to_minimum_vertex = {}
    new_vertices = []
    cnt_new_vertices = 0
    for vertex in vertices:
        vertex_key = tuple(vertex)
        
        if vertex_key not in coord_to_minimum_vertex.keys():
            coord_to_minimum_vertex[vertex_key] = cnt_new_vertices
            new_vertices.append(vertex)
            cnt_new_vertices += 1
    
    new_faces = []
    for face in faces_with_coord:
        new_faces.append([
            coord_to_minimum_vertex[coord] for coord in face
        ])
    
    return np.stack(new_vertices), np.array(new_faces, dtype=np.int32)


def reorder_vertices(vertices_df, columns=["z", "y", "x"], add_sorted_index=True):
    # sorting bottom -> top (descending order)
    vertices_df = vertices_df.sort_values(columns, ascending=False)
    
    if add_sorted_index:
        vertices_df["sorted_index"] = np.arange(len(vertices_df))
    return vertices_df


def reorder_faces(vertices_df, faces):
    # this func contains two sorting: in-face-triple sort and all-face sort.
    faces_sorted = []
    for face in faces:
        face_ids = vertices_df.loc[face]["sorted_index"]
        face_ids = np.sort(face_ids)
        faces_sorted.append(face_ids)
    
    # smaller index in face triple means nearer to bottom.
    faces_sorted = pd.DataFrame(faces_sorted)
    faces_sorted = faces_sorted.sort_values(list(range(3)), ascending=True)
    return pd.DataFrame(faces_sorted)


def bit_quantization(vertices, bit=8, v_min=-1., v_max=1.):
    # vertices must have values between -1 to 1.
    dynamic_range = 2 ** bit - 1
    discrete_interval = (v_max-v_min) / dynamic_range
    offset = (dynamic_range) / 2
    
    vertices = vertices / discrete_interval + offset
    vertices = np.clip(vertices, 0, dynamic_range-1)
    return vertices.astype(np.int32)
