import numpy as np
import cv2
import os
from alphapose_adapter import run_alphapose_thread
from preprocessing import preprocess_keypoints
from stgcn_model import run_stgcn_inference
# from alphapose_adapter import load_alphapose_model, run_alphapose_on_frames
import json
# At start of your script
cfg_path = r"C:\Users\FarehaIllyas\Desktop\AlphaPose_preproces\configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml"
ckpt_path = r"C:\Users\FarehaIllyas\Desktop\AlphaPose_preproces\AlphaPose\pretrained_models\fast_res50_256x192.pth"
# load_alphapose_model(cfg_path, ckpt_path)  # load model once
video_output_path = r"C:\Users\FarehaIllyas\Desktop\infrence_fall_detection\data"

"""Extracts all frames to out_folder; returns number of frames extracted."""
SLIDE = 5
def extract_frames_to_folder(video_path: str, out_folder: str, name: str, resize: tuple | None=None , window_size :int = 35  ) -> int:
    cap = cv2.VideoCapture(video_path )
    if not cap.isOpened() :
        print("Cannot open the video")
        return 0
    frame_count = 0
    buffer = FrameBuffer(window_size)
    success , frame = cap.read()
    
      # Prepare single video writer
    os.makedirs(out_folder, exist_ok=True)
    height, width, _ = frame.shape
    annotated_path = os.path.join(video_output_path, f"video_{name}.mp4")
    out = cv2.VideoWriter(
        annotated_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        25,
        (width, height)
    )


    while success:
        frame_name = f"frame_{frame_count:06d}.jpg"
        # frame_path = os.path.join(out_folder, frame_name)
        # cv2.imwrite(frame_path, frame)

        buffer.push(frame)
        frame_count += 1
        if(buffer.is_full()):
            # get buffer and do run alphapose on it to do extract skeleton points

            # separately store frames of one batch in a folder 
            temp_buffer_path = os.path.join(out_folder , f"batch_{frame_count//window_size}")
            os.makedirs(temp_buffer_path ,exist_ok=True )
            
            temp_frames_path = os.path.join(temp_buffer_path , "frames_output")
            os.makedirs(temp_frames_path ,exist_ok=True )
            
            for i , buff_frame in enumerate(buffer.get_window()):
                temp_frame_path = os.path.join(temp_frames_path , f"frame_{i:06d}.jpg")
                cv2.imwrite(temp_frame_path , buff_frame)

            # make a keypoints folder inside batch_1 folder to store keypoints output from alphapose
            keypoints_out = os.path.join(temp_buffer_path , "Keypoints_output")
            if len(os.listdir(temp_frames_path)) == 0:
                print(f"⚠️ No frames found in {temp_frames_path}, skipping AlphaPose.")
            else:
                print(f"path {temp_frames_path}")
                run_alphapose_thread(temp_frames_path, keypoints_out)
                # keypoints_results = run_alphapose_on_frames(buffer.get_window())
        
                os.makedirs(keypoints_out, exist_ok=True)

                frames_dir = os.path.join(keypoints_out, "vis")                
                json_path = os.path.join(keypoints_out , "alphapose-results.json")
                data_numpy = preprocess_keypoints(json_path ,frames_dir)      
                prediction = "No Person"
                if data_numpy is None:
                    print(f"No person detected in batch: {temp_buffer_path}, skipping model prediction.")
                else:
                    prediction = run_stgcn_inference(data_numpy)
                                        
                # for i, frame in enumerate(buffer.get_window()):
                #     cv2.putText(
                #         frame,
                #         f"Action: {prediction}",
                #         (50, 50),  # position
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         1,  # font scale
                #         (0, 0, 255),  # color BGR
                #         2,  # thickness
                #         cv2.LINE_AA
                #     )
                #     # Optional: show frame in window
                #     cv2.imshow("Real-Time Prediction", frame)
                #     cv2.waitKey(1)  # 1 ms delay for display

                    # save video
                    # annotated_path = os.path.join(temp_buffer_path, "annotated_batch.mp4")
                    # height, width, _ = buffer.get_window()[0].shape
                    # out = cv2.VideoWriter(
                    #     annotated_path, 
                    #     cv2.VideoWriter_fourcc(*'mp4v'), 
                    #     25,  # fps
                    #     (width, height)
                    # )

                    # for frame in buffer.get_window():
                    
                    for frame_name in sorted(os.listdir(frames_dir)):
                        frame_path = os.path.join(frames_dir, frame_name)
                        frame = cv2.imread(frame_path)

                        # Overlay prediction
                        cv2.putText(
                            frame,
                            f"Action: {prediction}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                        out.write(frame)  # write frame to video

                    # out.release()

            # buffer.pop_n(SLIDE)
            buffer.clear_buffer()
        success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return frame_count    

class FrameBuffer:
    def __init__(self, window_size: int):
        print("initilizing buffer")
        self.window_size = window_size
        self.buffer = []

    def push(self , frame: np.ndarray) -> None:
        # print("pushing a frame into buffer")
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(frame)

    def is_full(self) -> bool:
        if len(self.buffer) >= self.window_size:
            return True
        return False

    def get_window(self) -> list[np.ndarray]:
        print("getting a window of 35 frames")
        return self.buffer
    
    def clear_buffer(self)-> None:
        self.buffer = []

    def pop_n(self , k:int )-> None:
        self.buffer = self.buffer[k:]
