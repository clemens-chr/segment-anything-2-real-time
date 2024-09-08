from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import sam2
import tyro
from PIL import Image

from description_to_bbox import generate_bbox, generate_mask, visualize_results
from image_to_description import (
    generate_image_description,
)

# Import necessary functions
from mesh_to_image import generate_mesh_image

# ========================
# mesh_to_bbox function
# ========================


def mesh_to_bbox(
    image: Image.Image,
    mesh_filepath: Path,
) -> Tuple[Image.Image, str, np.ndarray, list, list]:
    """
    Goal: Get a bounding box of the object in the image based on the 3D mesh.
    How:
      1. Generate a mesh image from a mesh
      2. Generate a text description of the mesh using the OpenAI API
      3. Generate bounding boxes based on the description using the DINO model

    Args:
        image (Image.Image): The image to find bounding boxes for.
        mesh_filepath (Path): Path to the 3D mesh file.

    Returns:
        Tuple: (image, description, bboxes, confidences, class_names)
    """
    # Step 1: Generate image from mesh
    mesh_image = generate_mesh_image(
        mesh_filepath, image_width=300, image_height=300, radius=0.3
    )

    # Step 2: Generate a description of the image using OpenAI API
    description, _ = generate_image_description(
        image=mesh_image,
        openai_api_key=None,
        model="gpt-4o-mini",
        max_tokens=100,
    )

    # Step 3: Generate bounding boxes using the description as the text prompt
    # HACK: Add "on table" at the end of the description to improve grounding
    description += " on table"
    description = description.lower()
    print(f"HACK: Added 'on table' to description: {description}")

    bboxes, confidences, class_names = generate_bbox(
        image=image,
        text_prompt=description,
        grounding_model="gdino",
        gdino_1_5_api_token=None,
    )

    return mesh_image, description, bboxes, confidences, class_names


# ========================
# Main function and Tyro CLI
# ========================


@dataclass
class MeshToBBoxConfig:
    mesh_filepath: Path = (
        Path(__file__).parent / "example_data" / "woodblock_mesh" / "3DModel.obj"
    )
    image_path: Path = Path(__file__).parent / "example_data" / "woodblock.png"
    output_dir: Path = Path("outputs/mesh_to_bbox_demo")

    def __post_init__(self):
        assert self.mesh_filepath.exists(), f"{self.mesh_filepath} does not exist"
        assert self.image_path.exists(), f"{self.image_path} does not exist"
        self.output_dir.mkdir(parents=True, exist_ok=True)


def main():
    # Parse command-line arguments
    config = tyro.cli(MeshToBBoxConfig)

    image = Image.open(config.image_path)

    # Step 1: Run mesh_to_bbox to get image, description, bounding boxes, confidences, and class names
    mesh_image, description, bboxes, confidences, class_names = mesh_to_bbox(
        image=Image.open(config.image_path),
        mesh_filepath=config.mesh_filepath,
    )
    mesh_image_path = config.output_dir / "mesh_image.png"
    mesh_image.save(mesh_image_path)
    print(f"Mesh image saved to: {mesh_image_path}")
    print(f"Description: {description}")

    # Step 2: Generate mask using SAM2 based on the bounding boxes
    sam2_checkpoint: Path = (
        Path(sam2.__file__).parents[1] / "checkpoints/sam2_hiera_large.pt"
    )
    sam2_model_config: str = "sam2_hiera_l.yaml"
    assert sam2_checkpoint.exists(), f"{sam2_checkpoint} does not exist"
    masks, _, _ = generate_mask(
        image=image,
        sam2_model_config=sam2_model_config,
        sam2_checkpoint=sam2_checkpoint,
        bboxes=bboxes,
    )

    # Step 3: Visualize results and save to output directory
    visualize_results(
        image=image,
        bboxes=bboxes,
        confidences=confidences,
        class_names=class_names,
        masks=masks,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    main()
