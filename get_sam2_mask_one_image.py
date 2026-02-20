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

model_mask_path = "mask_sam2.1_hiera_large"
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

#subject = "female-3-casual"
#images = sorted(glob.glob(os.path.join("../../data/peoplesnapshot_arah-format/people_snapshot_public/",subject,"image/*")))
#os.makedirs(os.path.join("../../data/peoplesnapshot_arah-format/people_snapshot_public/",subject,model_mask_path),exist_ok=True)

image_path = sys.argv[1]
output_folder = sys.argv[2]
overlay_mask_folder = sys.argv[3]
smpl_folder = sys.argv[4]
camera_path = sys.argv[5]
kernel_size = int(sys.argv[6])


os.makedirs(output_folder,exist_ok=True)

with open(camera_path, 'rb') as f:
            camera = pkl.load(f, encoding='latin1')
K,_,_,distortion = get_KRTD(camera)


_,image_name = os.path.split(image_path)
image = Image.open(image_path)
image = np.array(image.convert("RGB"))
image = cv2.undistort(image,K,distortion,None)

mask_path = os.path.join(overlay_mask_folder,image_name[:-3]+"jpg")
mask = np.array(Image.open(mask_path))


smpl_file = os.path.join(smpl_folder,image_name[:-4]+".npz")
## load params
print(smpl_file)
params = np.load(smpl_file, allow_pickle=True)
abs_bone_transforms = np.array(params['abs_bone_transforms'])
trans = np.array(params['trans']).reshape([1, 3])
bones = K @ (abs_bone_transforms[1][0:3,3] + trans[0])
x = int(bones[0] / bones[2])
y = int(bones[1] / bones[2])

# compute box similiar to https://github.com/mikeqzy/3dgs-avatar-release/blob/main/train.py
#mask[mask == 255] = 0
mask_new = np.where(mask)
y1, y2 = mask_new[1].min(), mask_new[1].max() + 1
x1, x2 = mask_new[0].min(), mask_new[0].max() + 1
mask[x1:x2,y1:y2,:] = 100
input_point = np.array([[x,y]])
input_label = np.array([1])


# calculate intersection over union for each mask ref: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
boxArea = (x2-x1) * (y2-y1)
predictor.set_image(image)
masks, scores, logits = predictor.predict(
point_coords=input_point,
point_labels=input_label,
multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]
number_of_masks, w , h = masks.shape
iou = []
for i in range(number_of_masks):
        mask_new = np.where(masks[i])
        try:
                y1_tmp, y2_tmp = mask_new[1].min(), mask_new[1].max() + 1
                x1_tmp, x2_tmp = mask_new[0].min(), mask_new[0].max() + 1

                xA = max(x1_tmp,x1)
                yA = max(y1_tmp,y1)
                xB = min(x2_tmp,x2)
                yB = min(y2_tmp,y2)

                interArea = max(0, xB-xA+1)* max(0,yB-yA+1)
                boxTmpArea = (x2_tmp - x1_tmp) * (y2_tmp - y1_tmp)
                iou_score = interArea/ float(boxTmpArea +  boxArea - interArea)
                iou.append(iou_score)
                #if(0.4 * boxArea <= boxTmpArea and boxTmpArea <= 1.6 * boxArea):
                #       iou.append(iou_score)
                #else:
                #       iou.append(0)
        except:
                iou.append(0)

max_value = max(iou)
max_index = iou.index(max_value)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
masks[max_index] = cv2.erode(masks[max_index].copy(), kernel)
mask_result = np.stack((masks[max_index],masks[max_index],masks[max_index]),axis=2)
mask_result = (mask_result * 255).astype(np.uint8)
result = Image.fromarray(mask_result)
result.save(os.path.join(output_folder,"sam2.png"))
print(os.path.join(output_folder,"sam2.png"))



