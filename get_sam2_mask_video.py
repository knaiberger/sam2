from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
import os
import cv2
import time
import sys
import pickle as pkl
import matplotlib.pyplot as plt

# ref: https://github.com/mikeqzy/3dgs-avatar-release/blob/main/dataset/people_snapshot.py
def get_KRTD(camera):
        K = np.zeros([3, 3], dtype=np.float32)
        K[0, 0] = camera['camera_f'][0]
        K[1, 1] = camera['camera_f'][1]
        K[:2, 2] = camera['camera_c']
        K[2, 2] = 1
        R = np.eye(3, dtype=np.float32)
        T = np.zeros([3, 1], dtype=np.float32)
        D = camera['camera_k']

        return K, R, T, D

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

images_folder = sys.argv[1]
output_folder = sys.argv[2]
mask_path = sys.argv[3]
smpl_folder = sys.argv[4]
camera_path = sys.argv[5]
threshold = int(sys.argv[6])
kernel_size = int(sys.argv[7])



os.makedirs(output_folder,exist_ok=True)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(images_folder)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=images_folder)
predictor.reset_state(inference_state)
img = np.array(Image.open(mask_path))
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
count = 0
for i in range(len(img)):
	for j in range(len(img[i])):
		if(not img[i,j].any() == 0):
			count = count + 1
points = np.zeros((count,2))
labels = np.ones((count))
count = 0
for i in range(len(img)):
	for j in range(len(img[i])):
		if(not img[i,j].any() == 0):
			points[count,0] = i
			points[count,1] = j
			count = count + 1
			
		
print(points)
			
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

for out_frame_idx in range(0, len(frame_names)):
	for out_obj_id, out_mask in video_segments[out_frame_idx].items():
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		mask_result = np.stack((out_mask[0,:,:],out_mask[0,:,:],out_mask[0,:,:]),axis=2)
		mask_result = (mask_result * 255).astype(np.uint8)
		w,h,channel = mask_result.shape
		mask2 = np.zeros((w,h,channel)).astype(np.uint8)
		mask2[mask_result>=threshold] = 255
		mask_result = np.zeros((w,h,channel)).astype(np.uint8)
		mask_result[mask2!=255] = 255
		mask_result = cv2.erode(mask_result.copy(), kernel)
		mask_out_path = os.path.join(output_folder,"{:06d}.png".format(out_frame_idx))
		print(mask_out_path)
		result = Image.fromarray(mask_result)
		result.save(mask_out_path)
	






