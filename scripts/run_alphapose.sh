#!/bin/bash
# run_alphapose.sh
# Simple helper to run AlphaPose on a folder of frames

# Usage: ./run_alphapose.sh <input_frames_dir> <output_dir>

INPUT_DIR=$1
OUTPUT_DIR=$2

# Go to your AlphaPose repo path
cd C:\\Users\\FarehaIllyas\\Desktop\\AlphaPose_preproces

# Run alphapose
python3 scripts/demo_inference.py \
    --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained/pose_model.pth \
    --detector yolo \
    --indir "$INPUT_DIR" \
    --outdir "$OUTPUT_DIR" \
    --save_img

echo "AlphaPose finished processing frames from: $INPUT_DIR"
