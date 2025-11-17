import os
import json
import numpy as np
import cv2

TARGET_FRAMES = 35

# Joints to keep (13 joints)
SELECTED_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def preprocess_keypoints(json_path, frames_dir):
    # --- Load JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Get frame size ---
    first_frame = sorted(os.listdir(frames_dir))[0]
    img = cv2.imread(os.path.join(frames_dir, first_frame))
    H, W, _ = img.shape

    all_frames = []
    if len(data) == 0 :
        print("No person detected")
        return None
    # --- Process each frame ---
    for frame in data:
        kp = frame["keypoints"]
       
        x = np.array(kp[0::3])
        y = np.array(kp[1::3])
        c = np.array(kp[2::3])

        # Normalize to [0,1]
        x = x / W
        y = y / H

        # Center using left & right hip
        LHip, RHip = 11, 12
        if c[LHip] > 0 and c[RHip] > 0:
            cx = (x[LHip] + x[RHip]) / 2
            cy = (y[LHip] + y[RHip]) / 2
        else:
            cx = np.mean(x[c > 0])
            cy = np.mean(y[c > 0])

        x -= cx
        y -= cy

        # Stack joints → (17, 3)
        one_frame = np.stack([x, y, c], axis=1)

        # --- REMOVE eye & ear joints -> keep 13 joints ---
        one_frame = one_frame[SELECTED_JOINTS]   # (13, 3)

        all_frames.append(one_frame)

    # Convert to array (T, 13, 3)
    all_frames = np.array(all_frames)

    # --- Resample to 35 frames ---
    idx = np.linspace(0, len(all_frames)-1, TARGET_FRAMES).astype(int)
    all_frames = all_frames[idx]

    # ST-GCN format: (C, T, V, M)
    data = all_frames.transpose(2, 0, 1)    # (3, T, 13)
    # data = data[:, :, :, None]             # (3, T, 13, 1)
    data = data[:, :, :, None]  # (3, T, 13, 1) → keep person dim
    data = data[None, ...]      # (1, 3, T, 13, 1) → add batch dim


    return data
