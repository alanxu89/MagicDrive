# conda activate nuscenes

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/home/ubuntu/Documents/nuscenes-devkit/python-sdk")
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

NUSCENES_DATA_ROOT = '/home/ubuntu/Documents/ml_data/nuscenes_data/'
FPV_MAP_DIR = "fpv_semantic_maps"
SENSOR_LIST = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
LAYER_NAMES = [
    'drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
    'walkway', 'stop_line', 'carpark_area'
]

# scene
nusc = NuScenes(version='v1.0-trainval',
                dataroot=NUSCENES_DATA_ROOT,
                verbose=True)

# map
map_locations = [
    'boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth',
    'singapore-queenstown'
]
nusc_maps = dict()
for location in map_locations:
    nusc_maps[location] = NuScenesMap(dataroot=NUSCENES_DATA_ROOT,
                                      map_name=location)

# sensor dirs
for sensor in SENSOR_LIST:
    cam_dir = os.path.join(NUSCENES_DATA_ROOT, FPV_MAP_DIR, sensor)
    if not os.path.isdir(cam_dir):
        os.makedirs(cam_dir)

override = False
cnt = 0
# t0 = time.time()
# loop over scenes
for scene in nusc.scene:
    scene_token = scene['token']
    scene_rec = nusc.get('scene', scene_token)

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

    # get location and map
    log_record = nusc.get('log', scene_rec['log_token'])
    log_location = log_record['location']
    nusc_map = nusc_maps[log_location]

    sample_token = first_sample_token
    while sample_token:
        cnt += 1
        # print("sample token:", sample_token)
        sample_rec = nusc.get('sample', sample_token)

        for sensor in SENSOR_LIST:
            sd_rec = nusc.get('sample_data', sample_rec['data'][sensor])
            img_file = sd_rec['filename'].split('/')[-1]
            save_path = os.path.join(NUSCENES_DATA_ROOT, FPV_MAP_DIR, sensor,
                                     img_file)
            if (not os.path.exists(save_path)) or override:
                # t1 = time.time()
                nusc_map.render_map_in_image(nusc,
                                             sample_rec['token'],
                                             layer_names=LAYER_NAMES,
                                             camera_channel=sensor,
                                             alpha=1.0,
                                             render_dark_image=True,
                                             out_path=save_path)
                # print(time.time() - t1)
            else:
                print("pass")
                pass
            # plt.close(fig)

        # def func(sensor):
        #     sd_rec = nusc.get('sample_data', sample_rec['data'][sensor])
        #     img_file = sd_rec['filename'].split('/')[-1]
        #     save_path = os.path.join(NUSCENES_DATA_ROOT, FPV_MAP_DIR, sensor, img_file)
        #     nusc_map.render_map_in_image(nusc,
        #                     sample_rec['token'],
        #                     layer_names=LAYER_NAMES,
        #                     camera_channel=sensor,
        #                     alpha=1.0,
        #                     render_dark_image=True,
        #                     out_path=save_path)
        #     # plt.clf()
        # with Pool(len(SENSOR_LIST)) as p:
        #     p.map(func, SENSOR_LIST)

        sample_token = sample_rec['next']
    print(time.time() - t0)

    # break

# stop at scene 'c67b72f4a0bd4f2b96ecbd0e01c58232'. this scene has not started processing yet.
