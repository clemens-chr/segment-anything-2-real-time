#!/usr/bin/env python

from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header, Int32
from termcolor import colored
import pyrealsense2 as rs  # Add RealSense import

from mesh_to_bbox import generate_bbox, mesh_to_description
from sam2_model import SAM2Model


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    # PIL expects RGB, but OpenCV provides BGR
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def rgb_to_pil(rgb_image: np.ndarray) -> Image.Image:
    # PIL expects RGB
    return Image.fromarray(rgb_image)


def get_user_point(rgb_image: np.ndarray, title: str) -> Tuple[int, int]:
    # Get prompt as click
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.imshow(rgb_image)
    plt.axis("off")
    points = plt.ginput(1)  # get one click
    plt.close()

    x, y = int(points[0][0]), int(points[0][1])
    return x, y


def draw_prompts(image: np.ndarray, prompts: dict) -> np.ndarray:
    _H, _W, C = image.shape
    assert C == 3, f"{C}"
    image = image.copy()

    RED = [255, 0, 0]
    BLUE = [0, 0, 255]

    if prompts["box"] is not None:
        x_min, y_min, x_max, y_max = (
            int(prompts["box"][0]),
            int(prompts["box"][1]),
            int(prompts["box"][2]),
            int(prompts["box"][3]),
        )
        BOX_THICKNESS = 2
        BOX_COLOR = BLUE
        # Draw horizontal lines
        image[y_min : y_min + BOX_THICKNESS, x_min:x_max] = BOX_COLOR  # Top
        image[y_max - BOX_THICKNESS : y_max, x_min:x_max] = BOX_COLOR  # Bottom
        # Draw vertical lines
        image[y_min:y_max, x_min : x_min + BOX_THICKNESS] = BOX_COLOR  # Left
        image[y_min:y_max, x_max - BOX_THICKNESS : x_max] = BOX_COLOR  # Right

    if prompts["points"] is not None:
        points = prompts["points"]
        labels = prompts["labels"]
        N_points = points.shape[0]
        for i in range(N_points):
            point = points[i]
            label = labels[i]
            x, y = int(point[0]), int(point[1])
            POSITIVE_COLOR = BLUE
            NEGATIVE_COLOR = RED
            if label == 1:
                image[y - 5 : y + 5, x - 5 : x + 5] = POSITIVE_COLOR
            elif label == 0:
                image[y - 5 : y + 5, x - 5 : x + 5] = NEGATIVE_COLOR
            else:
                raise ValueError(f"Unknown label: {label}")
    return image


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

        # Add mask caching
        self.cached_mask = None
        self.use_mask_caching = rospy.get_param("~use_mask_caching", True)
        self.max_retries = rospy.get_param("~max_mask_retries", 50)
        self.current_retries = 0
        self.MIN_MASK_PIXELS = rospy.get_param("~min_mask_pixels", 100)  # Minimum pixels for a good mask

        # Try to initialize RealSense first
        self.use_realsense_api = False
        self.pipeline = None
        self.initialize_realsense()

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
        RATE_HZ = 10
        self.rate = rospy.Rate(RATE_HZ)
        print(colored("SAM2 ROS Node initialized and waiting for images...", "green"))

        # Cache generated text prompt to lower cost to generate bbox
        self.cached_generated_text_prompt: Optional[str] = None

    def initialize_realsense(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            
            # Warmup: Get first 10 frames to adjust exposure
            print(colored("Warming up RealSense camera (getting first 10 frames)...", "green"))
            for _ in range(10):
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("Failed to get color frame during warmup")
                rospy.sleep(0.1)  # Small delay between frames
            
            print(colored("Successfully connected to RealSense camera", "green"))
            self.use_realsense_api = True
        except Exception as e:
            print(colored(f"Could not connect to RealSense camera: {e}", "yellow"))
            self.use_realsense_api = False
            # Check camera parameter for ZED
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

    def image_callback(self, data):
        # Convert the ROS image message to a format OpenCV can work with
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def get_realsense_image(self):
        # Get frames from RealSense
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)  # Reduced timeout
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get color frame")
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            # Convert BGR to RGB
            self.rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(colored(f"Error getting RealSense frame: {e}", "red"))
            print(colored("Attempting to reconnect to RealSense...", "yellow"))
            self.initialize_realsense()
            return None

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

    def get_user_prompt(
        self,
        rgb_image: np.ndarray,
        use_negative_prompt: bool = False,
        use_2_points: bool = False,
    ):
        # Get prompt as click
        x, y = get_user_point(
            rgb_image=rgb_image, title="Click on the image to select a point"
        )
        print(f"Clicked point: ({x}, {y})")

        if use_negative_prompt:
            # Get negative prompt as click
            neg_x, neg_y = get_user_point(
                rgb_image=rgb_image,
                title="Click on the image to select a NEGATIVE point",
            )
            print(f"Clicked negative point: ({neg_x}, {neg_y})")

            points = np.array([[x, y], [neg_x, neg_y]], dtype=np.float32)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1, 0], dtype=np.int32)
        elif use_2_points:
            # Get a second prompt as click
            x_2, y_2 = get_user_point(
                rgb_image=rgb_image,
                title="Click on the image to select a SECOND point",
            )
            print(f"Clicked point: ({x_2}, {y_2})")

            points = np.array([[x, y], [x_2, y_2]], dtype=np.float32)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1, 1], dtype=np.int32)

        else:
            points = np.array([[x, y]], dtype=np.float32)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1], dtype=np.int32)

        return {
            "points": points,
            "labels": labels,
            "box": None,
        }

    def generate_sam_prompts(self, rgb_image: np.ndarray) -> Optional[dict]:
        PROMPT_METHOD: Literal[
            "mesh",
            "text",
            "hardcoded",
            "user_select",
            "user_select_with_negative",
            "user_select_with_2_points",
        ] = "user_select"  # CHANGE
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
        elif (
            PROMPT_METHOD == "user_select"
            or PROMPT_METHOD == "user_select_with_negative"
            or PROMPT_METHOD == "user_select_with_2_points"
        ):
            use_negative_prompt = PROMPT_METHOD == "user_select_with_negative"
            use_2_points = PROMPT_METHOD == "user_select_with_2_points"
            print(
                colored(
                    f"Using user select prompt (use_negative_prompt={use_negative_prompt}, use_2_points={use_2_points})",
                    "green",
                )
            )
            prompts = self.get_user_prompt(
                rgb_image=rgb_image,
                use_negative_prompt=use_negative_prompt,
                use_2_points=use_2_points,
            )
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

    def is_good_mask(self, mask: np.ndarray) -> bool:
        """Check if the mask is good enough."""
        num_mask_pixels = (mask[..., 0] > 0).sum()
        return num_mask_pixels > self.MIN_MASK_PIXELS

    def run(self):
        ##############################
        # Wait for the first image
        ##############################
        while not rospy.is_shutdown():
            if self.use_realsense_api:
                self.get_realsense_image()
            if self.rgb_image is not None:
                break
            print(colored("Waiting for the first image...", "green"))
            rospy.sleep(0.1)

        assert self.rgb_image is not None, "No image received"

        while not rospy.is_shutdown():
            if self.use_realsense_api:
                self.get_realsense_image()
                if self.rgb_image is None:  # If reconnection failed
                    continue
                
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
                    
                    if self.is_good_mask(mask):
                        self.cached_mask = mask.copy()
                        self.is_mask_initialized = True
                        print(colored("Initial mask is good, caching it", "green"))
                    else:
                        print(colored("Initial mask is not good enough, retrying...", "yellow"))
                        self.is_mask_initialized = False
            else:
                ##############################
                # Track
                ##############################
                start_time = rospy.Time.now()

                new_rgb_image = self.rgb_image.copy()

                mask = self.sam2_model.predict(
                    rgb_image=new_rgb_image, first=False, prompts=None
                )

                # Check if mask is terrible
                if not self.is_good_mask(mask):
                    print(
                        colored(
                            f"Mask is terrible, num_mask_pixels={(mask[..., 0] > 0).sum()}",
                            "yellow",
                        )
                    )
                    if self.use_mask_caching and self.cached_mask is not None:
                        print(colored("Using cached mask", "yellow"))
                        mask = self.cached_mask.copy()
                        self.current_retries += 1
                        
                        if self.current_retries >= self.max_retries:
                            print(colored("Max retries reached, reinitializing mask", "red"))
                            self.is_mask_initialized = False
                            self.current_retries = 0
                    else:
                        self.is_mask_initialized = False
                else:
                    print(
                        colored(
                            f"Mask is good, num_mask_pixels={(mask[..., 0] > 0).sum()}", "green"
                        )
                    )
                    if self.use_mask_caching:
                        self.cached_mask = mask.copy()
                        print(colored("Updated cached mask", "green"))
                    self.current_retries = 0
                    self.is_mask_initialized = True

                # Publish the number of mask pixels
                num_mask_pixels = (mask[..., 0] > 0).sum()
                self.num_mask_pixels_pub.publish(Int32(data=num_mask_pixels))

                # Convert OpenCV image (mask) to ROS Image message
                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="rgb8")
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
                    mask_rgb_with_prompt = draw_prompts(
                        image=mask.copy(), prompts=self.prompts
                    )

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

        # Clean up RealSense pipeline if using API directly
        if self.use_realsense_api and self.pipeline is not None:
            self.pipeline.stop()


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        node.run()
    except rospy.ROSInterruptException:
        print(colored("Shutting down SAM2 ROS Node.", "green"))
