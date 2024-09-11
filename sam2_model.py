import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2_camera_predictor

# Adapted from: github.com/Gy920/segment-anything-2-real-time/blob/main/demo/demo.py


class SAM2Model:
    def __init__(self):
        self.checkpoint = "checkpoints/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"

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

        self.predictor = build_sam2_camera_predictor(
            self.model_cfg, self.checkpoint, device=device
        )

    def predict(self, rgb_image, prompts=None, first=True) -> np.ndarray:
        # Make sure this is a np array of shape (H, W, 3) with RGB order of type np.uint8
        height, width, channels = rgb_image.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"
        assert rgb_image.dtype == np.uint8, f"Expected np.uint8, got {rgb_image.dtype}"

        mask_img = np.zeros((height, width, 3), dtype=np.uint8)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if first:
                self.predictor.load_first_frame(rgb_image)
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

                assert (
                    prompts is not None
                ), "prompts must be provided for the first frame"

                _, _, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=prompts["points"],
                    labels=prompts["labels"],
                    bbox=prompts["box"],
                )

            else:
                _, out_mask_logits = self.predictor.track(rgb_image)

            out_mask_logits = out_mask_logits.squeeze(dim=0).squeeze(dim=0)
            out_mask_logits = out_mask_logits.cpu().numpy()
            mask_img[out_mask_logits > 0] = [255, 255, 255]  # object is in white

            return mask_img

    def get_hardcoded_prompts(self):
        # Let's add a positive click at (x, y)
        points = np.array([[480, 440]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        point_labels = np.array([1], np.int32)

        # Need (N, 2) and (N,) shapes
        # Or both None
        if points is not None:
            assert (
                points.shape[0] == point_labels.shape[0]
            ), f"{points.shape}, {point_labels.shape}"

        # Let's add a box at (x_min, y_min, x_max, y_max)
        box = np.array([322, 415, 387, 480], dtype=np.float32)

        return {
            "points": points,
            "labels": point_labels,
            "box": box,
        }

    def visualize(self, rgb_image, prompts, out_mask_logits):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes = axes.flatten()

        # Image
        axes[0].imshow(rgb_image)
        self.show_points(prompts["points"], prompts["labels"], axes[0])
        self.show_box(prompts["box"], axes[0])
        axes[0].set_title("Input Image")

        # Mask
        self.show_mask(
            (out_mask_logits[0] > 0.0),
            axes[1],
        )
        self.show_points(prompts["points"], prompts["labels"], axes[1])
        self.show_box(prompts["box"], axes[1])
        axes[1].set_title("Predicted Mask")

    # Visualization Utils
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        if coords is None:
            return

        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    def show_box(self, box, ax):
        if box is None:
            return

        assert box.shape == (4,)
        x_min, y_min, x_max, y_max = box
        x0 = (x_min + x_max) / 2
        y0 = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
