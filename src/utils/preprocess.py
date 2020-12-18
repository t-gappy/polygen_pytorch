import numpy as np


def bit_quantization(vertices, bit=8, v_min=-1., v_max=1.):
    # vertices must have values between -1 to 1.
    dynamic_range = 2 ** bit - 1
    discrete_interval = (v_max-v_min) / (dynamic_range)#dynamic_range
    offset = (dynamic_range) / 2
    
    vertices = vertices / discrete_interval + offset
    vertices = np.clip(vertices, 0, dynamic_range-1)
    return vertices.astype(np.int32)


def redirect_same_vertices(vertices, faces):
    faces_with_coord = []
    for face in faces:
        faces_with_coord.append([[tuple(vertices[v_idx]), f_idx] for v_idx, f_idx in face])
    
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
        face = np.array([
            [coord_to_minimum_vertex[coord], f_idx] for coord, f_idx in face
        ])
        new_faces.append(face)
    
    return np.stack(new_vertices), new_faces


def reorder_vertices(vertices):
    indeces = np.lexsort(vertices.T[::-1])[::-1]
    return vertices[indeces], indeces


def reorder_faces(faces, sort_v_ids, pad_id=-1):
    # apply sorted vertice-id and sort in-face-triple values.
    
    faces_ids = []
    faces_sorted = []
    for f in faces:
        f = np.stack([
            np.concatenate([np.where(sort_v_ids==v_idx)[0], np.array([n_idx])])
            for v_idx, n_idx in f
        ])
        f_ids = f[:, 0]
        
        max_idx = np.argmax(f_ids)
        sort_ids = np.arange(len(f_ids))
        sort_ids = np.concatenate([
            sort_ids[max_idx:], sort_ids[:max_idx]
        ])
        faces_ids.append(f_ids[sort_ids])
        faces_sorted.append(f[sort_ids])
        
    # padding for lexical sorting.
    max_length = max([len(f) for f in faces_ids])
    faces_ids = np.array([
        np.concatenate([f, np.array([pad_id]*(max_length-len(f)))]) 
        for f in faces_ids
    ])
    
    # lexical sort over face triples.
    indeces = np.lexsort(faces_ids.T[::-1])[::-1]
    faces_sorted = [faces_sorted[idx] for idx in indeces]
    return faces_sorted
