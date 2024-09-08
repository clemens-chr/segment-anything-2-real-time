#!/usr/bin/env python

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header
from mesh_to_bbox import mesh_to_bbox
from description_to_bbox import generate_bbox
from PIL import Image
from pathlib import Path

# Assuming your SAM2Model is in a file named sam2_model
from sam2_model import SAM2Model


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    # PIL expects RGB, but OpenCV provides BGR
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


class SAM2RosNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("sam2_ros_node", anonymous=True)

        # Create an instance of SAM2Model
        self.sam2_model = SAM2Model()

        # Initialize the CvBridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.image_sub_topic = "/camera/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_topic, ROSImage, self.callback)

        # Publisher for the predicted mask
        QUEUE_SIZE = 1  # Always use the latest, okay to drop old messages
        self.mask_pub_topic = "/sam2_mask"
        self.mask_pub = rospy.Publisher(
            self.mask_pub_topic, ROSImage, queue_size=QUEUE_SIZE
        )

        # Frame count for distinguishing first frame
        self.frame_count = 0

        rospy.loginfo("SAM2 ROS Node initialized and waiting for images...")

    def callback(self, data):
        try:
            # Convert the ROS image message to a format OpenCV can work with
            bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo(f"Image received from {self.image_sub_topic}")

            first = self.frame_count == 0
            if first:
                # Predict the bounding box of the object to get a prompt
                USE_MESH = False
                pil_image = bgr_to_pil(bgr_image)
                if USE_MESH:
                    MESH_FILEPATH = Path(
                        # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/blueblock/3DModel.obj"
                        # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/snackbox/3DModel.obj"
                        # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/woodblock/3DModel.obj"
                        "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/cup_ycbv/textured.obj"
                    )
                    rospy.loginfo(f"Using mesh for prompt: {MESH_FILEPATH}")
                    assert MESH_FILEPATH.exists(), f"{MESH_FILEPATH}"
                    _, _, bboxes, _, _ = mesh_to_bbox(
                        image=pil_image,
                        mesh_filepath=MESH_FILEPATH,
                    )
                else:
                    TEXT_PROMPT = "red cup"
                    rospy.loginfo(f"Using text prompt for prompt: {TEXT_PROMPT}")
                    bboxes, _, _ = generate_bbox(
                        image=pil_image,
                        text_prompt=TEXT_PROMPT,
                        grounding_model="gdino",
                        gdino_1_5_api_token=None,
                    )

                points = np.stack(
                    [bboxes[:, 0] + bboxes[:, 2] / 2, bboxes[:, 1] + bboxes[:, 3] / 2],
                    axis=1,
                ).astype(np.float32)
                assert bboxes.shape == (1, 4), f"{bboxes.shape}"
                assert points.shape == (1, 2), f"{points.shape}"

                self.prompts = {
                    # points are in xy order
                    "points": points,  
                    # for labels, `1` means positive click and `0` means negative click
                    "labels": np.array([1], np.int32),
                    # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
                    "box": bboxes,
                    # "box": np.array([322, 415, 387, 480], dtype=np.float32)
                }

                # NOTE: points are directly associated with labels
                # points and labels both None or of shape (N, 2) and (N,)
                # box must be None or of shape (N, 4)
                # assert self.prompts["points"]

                # Predict the mask using SAM2Model
                mask = self.sam2_model.predict(
                    bgr_image,
                    first=True,
                    viz=False,
                    prompts=self.prompts,
                )
            else:
                mask = self.sam2_model.predict(bgr_image, first=False, viz=False)
            self.frame_count += 1

            # Convert mask to a format that can be published
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # HACK
            x_min, y_min, x_max, y_max = (
                int(self.prompts["box"][0, 0]),
                int(self.prompts["box"][0, 1]),
                int(self.prompts["box"][0, 2]),
                int(self.prompts["box"][0, 3]),
            )
            DRAW_BOX = False
            if DRAW_BOX:
                mask_rgb[y_min:y_max, x_min:x_max] = [255, 0, 0]
            else:
                x_mean, y_mean = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                mask_rgb[y_mean - 5 : y_mean + 5, x_mean - 5 : x_mean + 5] = [0, 0, 255]

            # Convert OpenCV image (mask) to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding="rgb8")
            mask_msg.header = Header(stamp=rospy.Time.now())

            # Publish the mask to the /sam2_mask topic
            self.mask_pub.publish(mask_msg)
            rospy.loginfo("Predicted mask published to /sam2_mask")

        except CvBridgeError as e:
            rospy.logerr(f"Could not convert ROS Image to OpenCV image: {e}")
        except Exception as e:
            rospy.logerr(f"Error during prediction or publishing: {e}")


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SAM2 ROS Node.")
