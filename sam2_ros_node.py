#!/usr/bin/env python

from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header

from description_to_bbox import generate_bbox
from mesh_to_bbox import mesh_to_bbox

# Assuming your SAM2Model is in a file named sam2_model
from sam2_model import SAM2Model


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    # PIL expects RGB, but OpenCV provides BGR
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def rgb_to_pil(rgb_image: np.ndarray) -> Image.Image:
    # PIL expects RGB
    return Image.fromarray(rgb_image)


class SAM2RosNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("sam2_ros_node", anonymous=True)

        # Create an instance of SAM2Model
        self.sam2_model = SAM2Model()

        # Frame count for distinguishing first frame
        self.frame_count = 0

        # Initialize the RGB image
        self.rgb_image: Optional[np.ndarray] = None

        # Initialize the CvBridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.image_sub_topic = "/camera/color/image_raw"
        self.image_sub = rospy.Subscriber(
            self.image_sub_topic, ROSImage, self.image_callback
        )

        # Publisher for the predicted mask
        QUEUE_SIZE = 1  # Always use the latest, okay to drop old messages
        self.mask_pub_topic = "/sam2_mask"
        self.mask_pub = rospy.Publisher(
            self.mask_pub_topic, ROSImage, queue_size=QUEUE_SIZE
        )
        self.mask_with_prompt_pub_topic = "/sam2_mask_with_prompt"
        self.mask_with_prompt_pub = rospy.Publisher(
            self.mask_with_prompt_pub_topic, ROSImage, queue_size=QUEUE_SIZE
        )

        rospy.loginfo("SAM2 ROS Node initialized and waiting for images...")

    def image_callback(self, data):
        # Convert the ROS image message to a format OpenCV can work with
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        rospy.logdebug(f"Image received from {self.image_sub_topic}")

    def run(self):
        ##############################
        # Wait for the first image
        ##############################
        while not rospy.is_shutdown() and self.rgb_image is None:
            rospy.loginfo("Waiting for the first image...")
            rospy.sleep(0.1)

        assert self.rgb_image is not None, "No image received"

        ##############################
        # Run first time
        ##############################
        first_rgb_image = self.rgb_image.copy()
        first_pil_image = rgb_to_pil(first_rgb_image)

        # Predict the bounding box of the object to get a prompt
        PROMPT_METHOD: Literal["mesh", "text", "hardcoded"] = "text"  # CHANGE
        if PROMPT_METHOD == "mesh":
            MESH_FILEPATH = Path(
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/blueblock/3DModel.obj"
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/snackbox/3DModel.obj"
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/woodblock/3DModel.obj"
                "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/cup_ycbv/textured.obj"
            )
            rospy.loginfo(f"Using mesh for prompt: {MESH_FILEPATH}")
            assert MESH_FILEPATH.exists(), f"{MESH_FILEPATH}"
            _, _, bboxes, _, _ = mesh_to_bbox(
                image=first_pil_image,
                mesh_filepath=MESH_FILEPATH,
            )
            assert bboxes.shape == (1, 4), f"{bboxes.shape}"
            self.prompts = {
                "points": None,
                "labels": None,
                "box": bboxes[0],
            }
        elif PROMPT_METHOD == "text":
            TEXT_PROMPT = "red cup"
            rospy.loginfo(f"Using text prompt for prompt: {TEXT_PROMPT}")
            bboxes, _, _ = generate_bbox(
                image=first_pil_image,
                text_prompt=TEXT_PROMPT,
                grounding_model="gdino",
                gdino_1_5_api_token=None,
            )
            assert bboxes.shape == (1, 4), f"{bboxes.shape}"
            self.prompts = {
                "points": None,
                "labels": None,
                "box": bboxes[0],
            }
        elif PROMPT_METHOD == "hardcoded":
            rospy.loginfo("Using hardcoded prompt")
            self.prompts = self.sam2_model.get_hardcoded_prompt()

        # NOTE: points are directly associated with labels
        # points and labels both None or of shape (N, 2) and (N,)
        # box must be None or of shape (N, 4)
        assert (self.prompts["points"] is None) == (
            self.prompts["labels"] is None
        ), f"{self.prompts}"
        if self.prompts["points"] is not None:
            N = self.prompts["points"].shape[0]
            assert self.prompts["points"].shape == (N, 2), f"{self.prompts}"
            assert self.prompts["labels"].shape == (N,), f"{self.prompts}"
        if self.prompts["box"] is not None:
            assert self.prompts["box"].shape == (4,), f"{self.prompts}"

        # Predict the mask using SAM2Model
        mask = self.sam2_model.predict(
            rgb_image=first_rgb_image,
            first=True,
            prompts=self.prompts,
        )
        self.sam2_model.visualize(
            rgb_image=first_rgb_image, prompts=self.prompts, out_mask_logits=mask
        )

        ##############################
        # Track
        ##############################
        while not rospy.is_shutdown():
            new_rgb_image = self.rgb_image.copy()

            mask = self.sam2_model.predict(
                rgb_image=new_rgb_image, first=False, prompts=None
            )
            self.frame_count += 1

            assert (
                mask.shape == new_rgb_image.shape
            ), f"{mask.shape} != {new_rgb_image.shape}"
            mask_rgb = mask

            # Convert OpenCV image (mask) to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding="rgb8")
            mask_msg.header = Header(stamp=rospy.Time.now())
            self.mask_pub.publish(mask_msg)

            rospy.loginfo("Predicted mask published to /sam2_mask")

            # Publish the mask with prompt to the /sam2_mask_with_prompt topic
            PUB_MASK_WITH_PROMPT = True
            if PUB_MASK_WITH_PROMPT:
                mask_rgb_with_prompt = mask_rgb.copy()

                # HACK: Draw on the mask
                x_min, y_min, x_max, y_max = (
                    int(self.prompts["box"][0]),
                    int(self.prompts["box"][1]),
                    int(self.prompts["box"][2]),
                    int(self.prompts["box"][3]),
                )
                DRAW_BOX = False
                if DRAW_BOX:
                    mask_rgb_with_prompt[y_min:y_max, x_min:x_max] = [255, 0, 0]
                else:
                    x_mean, y_mean = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                    mask_rgb_with_prompt[
                        y_mean - 5 : y_mean + 5, x_mean - 5 : x_mean + 5
                    ] = [0, 0, 255]

                # Convert OpenCV image (mask) to ROS Image message
                mask_with_prompt_msg = self.bridge.cv2_to_imgmsg(
                    mask_rgb_with_prompt, encoding="rgb8"
                )
                mask_with_prompt_msg.header = Header(stamp=rospy.Time.now())
                self.mask_with_prompt_pub.publish(mask_with_prompt_msg)


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SAM2 ROS Node.")
