# conda activate nuscenes

import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/Documents/nuscenes-devkit/python-sdk")
from nuscenes.nuscenes import NuScenes

NUSCENES_DATA_ROOT = '/home/ubuntu/Documents/ml_data/nuscenes_data/'
FPV_DEPTH_DIR = "fpv_depths"
FPV_MAP_DIR = "fpv_semantic_maps"
SENSOR_LIST = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

# scene
nusc = NuScenes(version='v1.0-trainval',
                dataroot=NUSCENES_DATA_ROOT,
                verbose=True)

csv_file = open(os.path.join(NUSCENES_DATA_ROOT, "data.csv"), mode="w")
csv_file.write("scene,timestamp,camera,mode,filepath,location,description\n")

for scene in nusc.scene:
    scene_token = scene['token']
    scene_rec = nusc.get('scene', scene_token)
    scene_name = scene_rec['name']
    description = scene_rec['description']
    log_record = nusc.get('log', scene_rec['log_token'])
    log_location = log_record['location']

    first_sample_token = scene_rec['first_sample_token']

    # check scene exist
    sample_rec = nusc.get('sample', first_sample_token)
    sd_rec = nusc.get('sample_data', sample_rec['data']["CAM_FRONT"])
    sensor_filepath = os.path.join(NUSCENES_DATA_ROOT, sd_rec['filename'])
    if not os.path.exists(sensor_filepath):
        # scene_not_exist, continue to next scene
        continue
    print("scene token:", scene_token)

    t0 = time.time()
    sample_token = first_sample_token
    scene_incomplete = False
    while sample_token:
        sample_rec = nusc.get('sample', sample_token)
        sample_ts = sample_rec['timestamp']
        for camera in SENSOR_LIST:
            sd_rec = nusc.get('sample_data', sample_rec['data'][camera])
            image_filepath = sd_rec['filename']
            fname = sd_rec['filename'].split('/')[-1]
            depth_filepath = os.path.join(FPV_DEPTH_DIR, camera, fname)
            map_filepath = os.path.join(FPV_MAP_DIR, camera, fname)
            abs_depth_filepath = os.path.join(NUSCENES_DATA_ROOT,
                                              depth_filepath)
            abs_map_filepath = os.path.join(NUSCENES_DATA_ROOT, map_filepath)
            if (not os.path.exists(abs_map_filepath)) or (
                    not os.path.exists(abs_depth_filepath)):
                scene_incomplete = True
                print("scene_incomplete, break")
                break

            for mode, fpath in zip(
                ["rgb", "depth", "map"],
                [image_filepath, depth_filepath, map_filepath]):
                csv_file.write(
                    f"{scene_name},{sample_ts},{camera},{mode},{fpath},{log_location},\"{description}\"\n"
                )
        if scene_incomplete:
            break

        sample_token = sample_rec['next']
    # break
csv_file.close()
