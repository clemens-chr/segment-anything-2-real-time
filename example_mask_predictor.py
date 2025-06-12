import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2_ros_node import MaskPredictor, draw_prompts

# Example: Load an image (replace with your own image path)
try:
    rgb_image = cv2.imread('example_image.jpg')
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
except Exception:
    # If no image, create a dummy image
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_image[100:300, 200:400] = [255, 0, 0]  # Add a red rectangle

# Initialize the mask predictor
predictor = MaskPredictor(min_mask_pixels=100, max_retries=10, use_mask_caching=True)

# Example text prompt
text_prompt = "red object"

# Predict the mask
try:
    mask = predictor.predict_mask(rgb_image, text_prompt=text_prompt)
    print("Mask shape:", mask.shape)

    # Visualize the mask
    plt.figure()
    plt.title("Predicted Mask")
    plt.imshow(mask)
    plt.axis('off')
    plt.show()

    # Visualize the mask with prompts
    if predictor.prompts is not None:
        mask_with_prompts = draw_prompts(mask.copy(), predictor.prompts)
        plt.figure()
        plt.title("Mask with Prompts")
        plt.imshow(mask_with_prompts)
        plt.axis('off')
        plt.show()
except Exception as e:
    print("Error during mask prediction:", e) 