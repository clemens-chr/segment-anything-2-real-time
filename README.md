# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream**

# TYLER DOCUMENTATION (September 8, 2024)

## CHANGES
Difference between the default SAM2 (https://github.com/facebookresearch/segment-anything-2) and real-time SAM2 (https://github.com/Gy920/segment-anything-2-real-time):

* Creates `sam2_camera_predictor.py`, which is nearly identical to `sam2_video_predictor.py`, but doesn't read in all frames at once from a file, but predict sequentially on new images

Difference between real-time SAM2 (https://github.com/Gy920/segment-anything-2-real-time) and this fork of real-time SAM2 (https://github.com/tylerlum/segment-anything-2-real-time):

* Slight modifications to `sam2_camera_predictor.py` to properly handle bounding box prompts

* Addition of `sam2_ros_node.py`, which listens for RGB images and outputs a mask. It needs a prompt, which can come from a text prompt, a mesh => image => text prompt, or a hardcoded position

* Addition of `sam2_model.py`, which is a nice wrapper around the `sam2_camera_predictor.py`. It is very robust, doesn't seem to need to re-start tracking extreme for extreme cases.

* Addition of `mesh_to_image.py` to go from mesh to mesh image (pyvista), `image_to_description.py` to go from mesh image to text description (GPT-4o), `description_to_bbox.py` to go from text description to bounding box around that object in a new image (Grounding DINO), and `mesh_to_bbox.py` which puts these things together. All of these are runnable scripts you can try.

## HOW TO RUN

### Install

ROS Noetic installation with Robostack (https://robostack.github.io/GettingStarted.html)
```
conda install mamba -c conda-forge
mamba create -n sam2_ros_env python=3.11
mamba activate sam2_ros_env

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

mamba install ros-noetic-desktop

mamba deactivate
mamba activate sam2_ros_env
```

Grounded SAM2 install (https://github.com/IDEA-Research/Grounded-SAM-2)
```
git clone https://github.com/IDEA-Research/Grounded-SAM-2
cd Grounded-SAM-2

cd checkpoints
bash download_ckpts.sh

cd ..
cd gdino_checkpoints
bash download_ckpts.sh

pip3 install torch torchvision torchaudio
export CUDA_HOME=/path/to/cuda-12.1/  # e.g., export CUDA_HOME=/usr/local/cuda-12.2

pip install -e .
pip install --no-build-isolation -e grounding_dino

pip install supervision pycocotools yapf timm 
pip install dds-cloudapi-sdk
pip install flash_attn einops

# Can also get Grounding DINO 1.5 API token if desired, refer to https://github.com/IDEA-Research/Grounded-SAM-2 for details
# I put my api tokens in
vim ~/api_keys/grounded_sam_2_key.txt
vim ~/api_keys/tml_openai_key.txt 
```

Then, run with:
```
python sam2_ros_node.py
```

# SUCCESSES AND FAILURES

## Florence

Was better than Grounding DINO for things like "primary object", but not as reliable in general. When given a good prompt, Grounding DINO seemed to be better.

<div align="center">
  <img src="https://github.com/user-attachments/assets/16be45ba-1360-411e-b81b-993cd4354066" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/8f9d1c0f-2b36-4129-9b0a-3841d67945bc" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/68997ed1-9221-4aeb-aca7-0af4b044229d" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/ca248dd5-54cc-45e4-8bba-eba2fb29ec63" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c6653ecf-80de-4e4e-8c6a-9f3b19a0b4c1" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/42dfab34-4f3a-4593-8630-56209cdf7b91" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/b12772f2-d715-4052-a319-c297a5b6f867" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/94d97b90-034b-43cf-9e51-78a0e08f3211" width="400" style="display: inline-block;" />
</div>

## Grounding DINO

Grounding DINO worked well with a good text prompt, but very poorly otherwise. This motivates the use of ChatGPT to caption the mesh image (automated pipeline without human).

<div align="center">
  <img src="https://github.com/user-attachments/assets/92087fba-02b8-4d6f-8807-0bee61971762" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/3e7ee901-92ca-45b8-abf9-0f1a7969498f" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/61c3cff7-b6af-4128-aa71-2d538d553a3f" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/a8e81c7a-e155-4fba-bb32-86aa9599f380" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/30f4cd1e-e87a-422d-9f66-48bdcbac519c" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/a53757cd-60e0-4370-a055-1ca5c296a11d" width="400" style="display: inline-block;" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/7eb15a18-04ec-4ebf-88ce-973216777ad7" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/d6c7dd79-8532-4426-8693-822d86ad75d1" width="400" style="display: inline-block;" />
</div>

## T-REX/DINO-V/OWL-VIT

T-REX and DINO-V and OWL-VIT can be conditioned on an image, but they were not reliable enough for me.

<div align="center">
  <img src="https://github.com/user-attachments/assets/8f02aac0-2293-45cb-bb0d-b4e8d50af981" width="400" style="display: inline-block;" />
</div>

https://huggingface.co/spaces/johko/image-guided-owlvit
<div align="center">
  <img src="https://github.com/user-attachments/assets/e4d92f08-8433-433a-a7d1-b0b4740e36fa" width="400" style="display: inline-block;" />
  <img src="https://github.com/user-attachments/assets/805b49df-508f-4cda-a9e3-c21450891d4b" width="400" style="display: inline-block;" />
</div>



# ORIGINAL DOCUMENTATION

## News
- 20/08/2024 : Fix management of ```non_cond_frame_outputs``` for better performance and add bbox prompt

## Demos
<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>

</div>



## Getting Started

### Installation

```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Camera prediction

```python
import torch
from sam2.build_sam import build_sam2_camera_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture(<your video or camera >)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(<your promot >)

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            ...
```

## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
