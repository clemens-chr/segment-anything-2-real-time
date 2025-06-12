import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time
from mask_predictor import MaskPredictor

# Parameters
NUM_FRAMES = 1000
TEXT_PROMPT = "object"  # Change as needed
PRINT_ONLY_MASK_PIXELS = True  # Set to True to only print mask pixel count, False to plot

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Helper to get a valid frame, with reconnection logic
def get_valid_frame(pipeline, config):
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            if color_frame:
                return color_frame
            else:
                raise RuntimeError("No color frame received.")
        except Exception as e:
            print("Error getting frame from RealSense:", e)
            print("Attempting to reconnect RealSense pipeline...")
            try:
                pipeline.stop()
            except Exception:
                pass
            time.sleep(1)
            pipeline.start(config)
            time.sleep(1)

try:
    pipeline.start(config)
    print("RealSense pipeline started.")

    # Warmup: Get first 10 frames to adjust exposure
    for _ in range(10):
        color_frame = get_valid_frame(pipeline, config)
        time.sleep(0.1)

    # Initialize the mask predictor
    predictor = MaskPredictor()
    mask = None

    if not PRINT_ONLY_MASK_PIXELS:
        plt.ion()
        fig, ax = plt.subplots()
        img_disp = None

    for i in range(NUM_FRAMES):
        color_frame = get_valid_frame(pipeline, config)
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        
        mask, num_mask_pixels = predictor.predict_mask(rgb_image)


        if not PRINT_ONLY_MASK_PIXELS:
            if img_disp is None:
                img_disp = ax.imshow(mask)
                plt.title("Predicted Mask (Live)")
                plt.axis('off')
            else:
                img_disp.set_data(mask)
            plt.pause(0.001)

    if not PRINT_ONLY_MASK_PIXELS:
        plt.ioff()
        plt.show()
    pipeline.stop()
    print("RealSense pipeline stopped.")

except Exception as e:
    print("Error with RealSense or mask prediction:", e)
    try:
        pipeline.stop()
    except Exception:
        pass 