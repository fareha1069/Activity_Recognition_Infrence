import sys
sys.path.insert(0, r"C:\Users\FarehaIllyas\Desktop\Pose_Based_Fall_Detection")


import numpy as np
import torch

# ----- Add path to your ST-GCN repo -----
from net.st_gcn import Model
from net.utils.graph import Graph

def run_stgcn_inference(arr):
    print("its not working")
    # ----- Load preprocessed keypoints -----
    # (Shape expected: [1, 3, T, 13, 1])
    # keypoints_path = r"C9:\Users\FarehaIllyas\Desktop\AlphaPose_preproces\output\falls\2\clean_keypoints.npy"
    # arr = np.load(keypoints_path)

    x = torch.tensor(arr, dtype=torch.float32)

    # ----- Load trained ST-GCN model -----
    graph_args = {'layout': 'mydataset', 'strategy': 'spatial'}

    model = Model(
        in_channels=3,
        num_class=7,
        graph_args=graph_args,
        edge_importance_weighting=True
    )
    model.load_state_dict(
        torch.load(
            r"C:\Users\FarehaIllyas\Desktop\Pose_Based_Fall_Detection\work_dir\alphapose-13\train\epoch30_model.pt",
            # r"C:\Users\FarehaIllyas\Desktop\Pose_Based_Fall_Detection\work_dir\fall_detection_system\train\epoch50_model.pt",
            map_location='cpu'
        )
    )
    model.eval()

    # ----- Predict -----
    actions = ['walking', 'sitting', 'standing', 'sit down', 'stand up', 'lying', 'fall down']
    
    # actions = ['not fall', 'fall down']

    with torch.no_grad():
        output = model(x)
        pred_class = torch.argmax(output, dim=1)
        print("Predicted action:", actions[pred_class.item()])
        return actions[pred_class.item()]
