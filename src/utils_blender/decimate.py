# code for blender 2.92.0
import os
import bpy
import math
import random


ANGLE_MIN = 1
ANGLE_MAX = 20
RESIZE_MIN = 0.75
RESIZE_MAX = 1.25
N_V_MAX = 800
N_F_MAX = 2800
NUM_AUGMENT = 30
AUG_TRY_MAX = 50
SEPARATOR = "/"
PATH_TEXT = "PATH_TO_DATAPATH_TEXT"
TEMP_PATH = "PATH_TO_TEMP_FILE"
OUT_DIR = "PATH_TO_OUT_DIR" + SEPARATOR + "{}" + SEPARATOR + "{}"
OBJ_NAME = "model_normalized"



def delete_scene_objects():
    scene = bpy.context.scene
            
    for object_ in scene.objects:
        bpy.data.objects.remove(object_)

    

def load_obj(filepath):
    bpy.ops.import_scene.obj(filepath=filepath)
    


def create_rand_scale(min, max):
    return [random.uniform(min, max) for i in range(3)]


def resize(scale_vec):
    bpy.ops.transform.resize(value=scale_vec, constraint_axis=(True,True,True))


def decimate(angle_limit=5):
    bpy.ops.object.modifier_add(type='DECIMATE')
    decim = bpy.context.object.modifiers["デシメート"]
    decim.decimate_type = 'DISSOLVE'
    decim.delimit = {'MATERIAL'}
    angle_limit_pi = angle_limit / 180 * math.pi
    decim.angle_limit = angle_limit_pi



if __name__ == "__main__":
    
    paths = []
    with open(PATH_TEXT) as fr:
        for line in fr:
            paths.append(line.rstrip().split("\t"))
    
    
    
    last_tag = ""
    
    for tag, path in paths:
        cnt_cleared = 0
        cnt_not_cleared = 0
        if last_tag != tag:
            last_tag = tag
            num_augment_ended = 0
            
        now_out_dir = OUT_DIR.format(tag.split(",")[0], str(num_augment_ended))
        os.makedirs(now_out_dir, exist_ok=True)
        
        
        while cnt_cleared < NUM_AUGMENT:
            
            if cnt_not_cleared > NUM_AUGMENT:
                break
            
            # delete all objects before loading.
            delete_scene_objects()
            
            # load .obj file
            load_obj(path)
            
            # search object key to decimate.
            for k in bpy.data.objects.keys():
                if OBJ_NAME in k:
                    obj_key = k
            
            # select object to be decimated.
            bpy.context.view_layer.objects.active = bpy.data.objects[obj_key]
            
            # setting parameters for preprocess.
            angle_limit = random.randrange(ANGLE_MIN, ANGLE_MAX)
            resize_scales = create_rand_scale(RESIZE_MIN, RESIZE_MAX)
            
            # perform preprocesses.
            decimate(angle_limit=angle_limit)
            resize(resize_scales)
    
            # save as temporary file.
            bpy.ops.export_scene.obj(filepath=TEMP_PATH)
    
            # check saving threshold.
            with open(TEMP_PATH) as fr:
                texts = [l.rstrip() for l in fr]
            n_vertices = len([l for l in texts if l[:2] == "v "])
            n_faces = len([l for l in texts if l[:2] == "f "])
    
            if (n_vertices <= N_V_MAX) and (n_faces <= N_F_MAX):
                out_name = "decimate_{}_scale_{:.5f}_{:.5f}_{:.5f}".format(angle_limit, *resize_scales)
                out_path = now_out_dir + SEPARATOR + out_name
                bpy.ops.export_scene.obj(filepath=out_path)
                cnt_cleared += 1
            else:
                cnt_not_cleared += 1
        
        num_augment_ended += 1
                
