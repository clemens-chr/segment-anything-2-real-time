#!/usr/bin/env python

from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header, Int32
from termcolor import colored

from mesh_to_bbox import generate_bbox, mesh_to_description
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

        # Latest RGB image
        self.rgb_image: Optional[np.ndarray] = None

        # State to check if mask is already initialized (only tracking needed)
        self.is_mask_initialized = False

        # Initialize the CvBridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()

        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            camera = DEFAULT_CAMERA
        print(colored(f"Using camera: {camera}", "green"))
        if camera == "zed":
            self.image_sub_topic = "/zed/zed_node/rgb/image_rect_color"
        elif camera == "realsense":
            self.image_sub_topic = "/camera/color/image_raw"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribe to the camera topic
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

        # Publisher for num_mask_pixels
        self.num_mask_pixels_pub_topic = "/sam2_num_mask_pixels"
        self.num_mask_pixels_pub = rospy.Publisher(
            self.num_mask_pixels_pub_topic, Int32, queue_size=QUEUE_SIZE
        )

        self.reset_sub = rospy.Subscriber(
            "/sam2_reset", Int32, self.reset_callback, queue_size=1
        )

        # Rate
        RATE_HZ = 1
        self.rate = rospy.Rate(RATE_HZ)
        print(colored("SAM2 ROS Node initialized and waiting for images...", "green"))

        # Cache generated text prompt to lower cost to generate bbox
        self.cached_generated_text_prompt: Optional[str] = None

    def image_callback(self, data):
        # Convert the ROS image message to a format OpenCV can work with
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def reset_callback(self, data):
        if data.data > 0:
            print(colored("Resetting the sam2 node", "green"))
            self.is_mask_initialized = False
        else:
            print(colored("Received a reset message with data <= 0", "green"))

    def generate_sam_prompts_from_mesh(
        self, rgb_image: np.ndarray, mesh_filepath: Path
    ) -> Optional[dict]:
        if self.cached_generated_text_prompt is not None:
            print(
                colored(
                    f"Using cached generated text prompt: {self.cached_generated_text_prompt}",
                    "green",
                )
            )
        else:
            print(
                colored(
                    "No cached generated text prompt, generating new text prompt...",
                    "green",
                )
            )

            # Use mesh to predict the bounding box of the object to get a prompt
            print(colored(f"Using mesh for prompt: {mesh_filepath}", "green"))
            assert mesh_filepath.exists(), f"{mesh_filepath}"
            _, generated_text_prompt = mesh_to_description(
                mesh_filepath=mesh_filepath,
            )
            print(colored(f"Generated text prompt: {generated_text_prompt}", "green"))
            self.cached_generated_text_prompt = generated_text_prompt

        return self.generate_sam_prompts_from_text(
            rgb_image=rgb_image, text_prompt=self.cached_generated_text_prompt
        )

    def generate_sam_prompts_from_text(
        self, rgb_image: np.ndarray, text_prompt: str
    ) -> Optional[dict]:
        pil_image = rgb_to_pil(rgb_image)

        try:
            bboxes, _, _ = generate_bbox(
                image=pil_image,
                text_prompt=text_prompt,
                grounding_model="gdino",
                gdino_1_5_api_token=None,
            )
            assert bboxes.shape == (1, 4), f"{bboxes.shape}"
            return {
                "points": None,
                "labels": None,
                "box": bboxes[0],
            }
        except ValueError as e:
            print(colored(f"Error: {e}", "red"))
            print(colored("No object found in the image using text prompt", "red"))
            return None

    def generate_sam_prompts(self, rgb_image: np.ndarray) -> Optional[dict]:
        PROMPT_METHOD: Literal["mesh", "text", "hardcoded"] = "text"  # CHANGE
        print(colored(f"Using prompt method: {PROMPT_METHOD}", "green"))

        if PROMPT_METHOD == "mesh":
            mesh_file = rospy.get_param("/mesh_file", None)
            if mesh_file is None:
                DEFAULT_MESH_FILEPATH = Path(
                    # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/blueblock/3DModel.obj"
                    # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/snackbox/3DModel.obj"
                    # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/woodblock/3DModel.obj"
                    "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/cup_ycbv/textured.obj"
                )
                mesh_file = str(DEFAULT_MESH_FILEPATH)
                print(colored(f"Using default mesh file: {mesh_file}", "yellow"))
            print(colored(f"Using mesh for prompt: {mesh_file}", "green"))

            prompts = self.generate_sam_prompts_from_mesh(
                rgb_image=rgb_image, mesh_filepath=Path(mesh_file)
            )
        elif PROMPT_METHOD == "text":
            text_prompt = rospy.get_param("/text_prompt", None)
            if text_prompt is None:
                DEFAULT_TEXT_PROMPT = "red cup"
                # DEFAULT_TEXT_PROMPT = "red cracker box"

                text_prompt = DEFAULT_TEXT_PROMPT
                print(colored(f"Using default text prompt: {text_prompt}", "yellow"))

            print(colored(f"Using text prompt for prompt: {text_prompt}", "green"))
            prompts = self.generate_sam_prompts_from_text(
                rgb_image=rgb_image, text_prompt=text_prompt
            )
        elif PROMPT_METHOD == "hardcoded":
            print(colored("Using hardcoded prompt", "green"))
            prompts = self.sam2_model.get_hardcoded_prompts()
        else:
            raise ValueError(f"Unknown PROMPT_METHOD: {PROMPT_METHOD}")

        if prompts is not None:
            self.validate_sam_prompts(prompts)

        return prompts

    @staticmethod
    def validate_sam_prompts(prompts: dict) -> Optional[dict]:
        # NOTE: points are directly associated with labels
        # points and labels both None or of shape (N, 2) and (N,)
        # box must be None or of shape (N, 4)
        assert (prompts["points"] is None) == (prompts["labels"] is None), f"{prompts}"
        if prompts["points"] is not None:
            N = prompts["points"].shape[0]
            assert prompts["points"].shape == (N, 2), f"{prompts}"
            assert prompts["labels"].shape == (N,), f"{prompts}"
        if prompts["box"] is not None:
            assert prompts["box"].shape == (4,), f"{prompts}"

        return prompts

    def run(self):
        ##############################
        # Wait for the first image
        ##############################
        while not rospy.is_shutdown() and self.rgb_image is None:
            print(colored("Waiting for the first image...", "green"))
            rospy.sleep(0.1)

        assert self.rgb_image is not None, "No image received"

        while not rospy.is_shutdown():
            if not self.is_mask_initialized:
                ##############################
                # Run first time
                ##############################
                first_rgb_image = self.rgb_image.copy()
                self.prompts = self.generate_sam_prompts(rgb_image=first_rgb_image)

                if self.prompts is None:
                    print(
                        colored(
                            "Error: prompts is None. Likely means no object found in the image.",
                            "red",
                        )
                    )
                    WAIT_TIME_SECONDS = 0.5
                    print(
                        colored(
                            f"Waiting for {WAIT_TIME_SECONDS} seconds before trying again...",
                            "red",
                        )
                    )
                    rospy.sleep(WAIT_TIME_SECONDS)
                    self.is_mask_initialized = False
                else:
                    # Predict the mask using SAM2Model
                    mask = self.sam2_model.predict(
                        rgb_image=first_rgb_image,
                        first=True,
                        prompts=self.prompts,
                    )
                    self.is_mask_initialized = True
            else:
                ##############################
                # Track
                ##############################
                start_time = rospy.Time.now()

                new_rgb_image = self.rgb_image.copy()

                mask = self.sam2_model.predict(
                    rgb_image=new_rgb_image, first=False, prompts=None
                )

                assert mask.shape == new_rgb_image.shape, (
                    f"{mask.shape} != {new_rgb_image.shape}"
                )
                mask_rgb = mask

                # Check if mask is terrible
                num_mask_pixels = (mask_rgb[..., 0] > 0).sum()
                MIN_MASK_PIXELS = 0
                if num_mask_pixels <= MIN_MASK_PIXELS:
                    print(
                        colored(
                            f"Mask is terrible, num_mask_pixels={num_mask_pixels}",
                            "yellow",
                        )
                    )
                    self.is_mask_initialized = False
                else:
                    print(
                        colored(
                            f"Mask is good, num_mask_pixels={num_mask_pixels}", "green"
                        )
                    )
                    self.is_mask_initialized = True

                # Publish the number of mask pixels
                self.num_mask_pixels_pub.publish(Int32(data=num_mask_pixels))

                # Convert OpenCV image (mask) to ROS Image message
                mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding="rgb8")
                mask_msg.header = Header(stamp=rospy.Time.now())
                self.mask_pub.publish(mask_msg)

                print(colored("Predicted mask published to /sam2_mask", "green"))

                # Publish the mask with prompt
                PUB_MASK_WITH_PROMPT = True
                if self.prompts is None:
                    print(
                        colored(
                            "prompts is None, skipping mask_with_prompt_pub", "yellow"
                        )
                    )
                if PUB_MASK_WITH_PROMPT and self.prompts is not None:
                    mask_rgb_with_prompt = mask_rgb.copy()

                    # HACK: Draw on the mask
                    x_min, y_min, x_max, y_max = (
                        int(self.prompts["box"][0]),
                        int(self.prompts["box"][1]),
                        int(self.prompts["box"][2]),
                        int(self.prompts["box"][3]),
                    )
                    DRAW_BOX = True
                    BOX_THICKNESS = 2
                    if DRAW_BOX:
                        # Draw horizontal lines
                        mask_rgb_with_prompt[
                            y_min : y_min + BOX_THICKNESS, x_min:x_max
                        ] = [255, 0, 0]  # Top
                        mask_rgb_with_prompt[
                            y_max - BOX_THICKNESS : y_max, x_min:x_max
                        ] = [255, 0, 0]  # Bottom
                        # Draw vertical lines
                        mask_rgb_with_prompt[
                            y_min:y_max, x_min : x_min + BOX_THICKNESS
                        ] = [255, 0, 0]  # Left
                        mask_rgb_with_prompt[
                            y_min:y_max, x_max - BOX_THICKNESS : x_max
                        ] = [255, 0, 0]  # Right
                    else:
                        x_mean, y_mean = (
                            int((x_min + x_max) / 2),
                            int((y_min + y_max) / 2),
                        )
                        mask_rgb_with_prompt[
                            y_mean - 5 : y_mean + 5, x_mean - 5 : x_mean + 5
                        ] = [0, 0, 255]

                    # Convert OpenCV image (mask) to ROS Image message
                    mask_with_prompt_msg = self.bridge.cv2_to_imgmsg(
                        mask_rgb_with_prompt, encoding="rgb8"
                    )
                    mask_with_prompt_msg.header = Header(stamp=rospy.Time.now())
                    self.mask_with_prompt_pub.publish(mask_with_prompt_msg)

                done_time = rospy.Time.now()
                self.rate.sleep()
                after_sleep_time = rospy.Time.now()
                print(
                    colored(
                        f"Max rate: {np.round(1.0 / (done_time - start_time).to_sec())} Hz ({np.round((done_time - start_time).to_sec() * 1000)} ms), Actual rate with sleep: {np.round(1.0 / (after_sleep_time - start_time).to_sec())} Hz",
                        "green",
                    )
                )


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        node.run()
    except rospy.ROSInterruptException:
        print(colored("Shutting down SAM2 ROS Node.", "green"))
