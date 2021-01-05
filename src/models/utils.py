import json
import torch


class Config(object):
    
    def write_to_json(self, out_path):
        with open(out_path, "w") as fw:
            json.dump(self.config, fw, indent=4)
            
    def load_from_json(self, file_path):
        with open(file_path) as fr:
            self.config = json.load(fr)
        
    def __getitem__(self, key):
        return self.config[key]
    
    
    
def accuracy(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=1)

    if ignore_label:
        normalizer = torch.sum(y_true!=ignore_label)
        ignore_mask = torch.where(
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)

    acc = (y_pred.reshape(-1)==y_true.reshape(-1)).type(torch.float32)
    acc = torch.sum(acc*ignore_mask)
    return acc / normalizer


class VertexDataset(torch.utils.data.Dataset):
    
    def __init__(self, vertices):
        self.vertices = vertices

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, idx):
        x = self.vertices[idx]
        return x
    
    
class FaceDataset(torch.utils.data.Dataset):
    
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, idx):
        x = self.vertices[idx]
        y = self.faces[idx]
        return x, y
    
    
def collate_fn_vertex(batch):
    return [{"vertices": batch}]


def collate_fn_face(batch):
    vertices = [d[0] for d in batch]
    faces = [d[1] for d in batch]
    return [{"vertices": vertices, "faces": faces}]