from video_io import extract_frames_to_folder
from preprocessing import preprocess_keypoints
from stgcn_model import run_stgcn_inference

video = "video_5.avi"
root_path = r"C:\Users\FarehaIllyas\Desktop\infrence_fall_detection\data\video_5"
extract_frames_to_folder(video , root_path , "video_5")
