import os

import sys
sys.path.insert(0, r"C:\Users\FarehaIllyas\Desktop\AlphaPose_preproces\AlphaPose")

import subprocess
import argparse
import yaml
import threading

def run_alphapose_thread(input_dir, output_dir):
    t = threading.Thread(target = run_alphapose , args=(input_dir , output_dir))
    t.start()
    t.join()

def run_alphapose(input_dir, output_dir):
    # Load config file for paths
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    python_path = config["alphapose"]["python_path"]
    alphapose_script = config["alphapose"]["demo_script"]
    cfg_path = config["alphapose"]["cfg"]
    ckpt_path = config["alphapose"]["checkpoint"]

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        python_path,
        alphapose_script,
        "--cfg", cfg_path,
        "--checkpoint", ckpt_path,
        "--detector", "yolo",
        "--indir", input_dir,
        "--outdir", output_dir,
        "--save_img"
    ]

    print(f"➡️ Running AlphaPose on {input_dir}")
    # subprocess.run(cmd)
    cwd = os.path.join(config["alphapose_repo"], "AlphaPose")
    subprocess.run(cmd, check=True, cwd=cwd)

    print(f"✅ Done. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

#     run_alphapose(args.indir, args.outdir)
# import torch
# from alphapose.models.builder import build_sppe
# from alphapose.utils.config import update_config
# from alphapose.utils.presets import SimpleTransform
# from alphapose.datasets.dataset_factory import get_dataset

# # Global model and config (loaded once)
# MODEL = None
# CFG = None
# TRANSFORM = None
# dataset_cfg = cfg.DATA_PRESET
# dataset = get_dataset(cfg.DATASET.TEST_DS)  # usually 'coco', 'mpii', etc.

# def load_alphapose_model(cfg_path, ckpt_path):
#     global MODEL, CFG, TRANSFORM
#     CFG = update_config(cfg_path)
#     MODEL = build_sppe(CFG.MODEL, preset_cfg=CFG.DATA_PRESET)
#     MODEL.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#     MODEL.eval()

#     TRANSFORM   = SimpleTransform(
#     dataset,                    # dataset object
#     add_dpg=False,
#     input_size=cfg.DATA_PRESET['IMAGE_SIZE'],
#     output_size=cfg.DATA_PRESET['HEATMAP_SIZE'],
#     scale_factor=0.25,
#     rot=0,
#     sigma=2,
#     train=False
# )

# def run_alphapose_on_frames(frames):
#     """frames: list of np.ndarray"""
#     results = []
#     global MODEL, TRANSFORM
#     for frame in frames:
#         img, center, scale = TRANSFORM.test_transform(
#             frame, [frame.shape[1]//2, frame.shape[0]//2], frame.shape[0]
#         )
#         img = torch.from_numpy(img).unsqueeze(0)  # add batch dim
#         with torch.no_grad():
#             pose = MODEL(img)
#         results.append(pose.cpu().numpy())
#     return results
