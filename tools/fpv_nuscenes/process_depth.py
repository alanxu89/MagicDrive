# conda activate opensora

from transformers import pipeline
from PIL import Image
import os
import time

BATCH_SIZE = 8
pipe = pipeline(task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device='cuda',
                batch_size=BATCH_SIZE)

NUSCENES_DATA_ROOT = '/home/ubuntu/Documents/ml_data/nuscenes_data/'
SAMPLES_DIR = "samples"
FPV_MAP_DIR = "fpv_depths"
CAMERA_LIST = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
# CAMERA_LIST = [
#     "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
#     "CAM_BACK", "CAM_BACK_LEFT"
# ]

override = False

for cam in CAMERA_LIST:
    save_dir = os.path.join(NUSCENES_DATA_ROOT, FPV_MAP_DIR, cam)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    cam_dir = os.path.join(NUSCENES_DATA_ROOT, SAMPLES_DIR, cam)
    jpg_files = os.listdir(cam_dir)
    jpg_files.sort()

    for i in range(0, len(jpg_files), BATCH_SIZE):
        image_file_batch = jpg_files[i:i + BATCH_SIZE]
        image_batch = [
            Image.open(os.path.join(cam_dir, fname))
            for fname in image_file_batch
        ]
        t0 = time.time()
        preds = pipe(image_batch)
        print(time.time() - t0)

        for j in range(len(image_batch)):
            fname = image_file_batch[j].split('/')[-1]
            save_path = os.path.join(save_dir, fname)

            if (not os.path.exists(save_path)) or override:
                depth = preds[j]["depth"]
                depth.save(save_path)

        # if i > 10 * BATCH_SIZE:
        #     break
    # break
