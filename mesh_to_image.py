from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
import tyro
from PIL import Image


@dataclass
class MeshToImage:
    mesh_filepath: Path = (
        Path(__file__).parent / "example_data" / "woodblock_mesh" / "3DModel.obj"
    )
    image_width: int = 300
    image_height: int = 300
    radius: float = 0.3
    output_image_path: Path = (
        Path(__file__).parent / "outputs" / "rendered_mesh_with_white_background.png"
    )

    def __post_init__(self) -> None:
        assert self.mesh_filepath.exists(), f"{self.mesh_filepath} does not exist"


def generate_mesh_image(
    mesh_filepath: Path, image_width: int, image_height: int, radius: float
) -> Image.Image:
    """
    Generate an image of the 3D mesh and return a PIL Image object.

    Args:
        mesh_filepath (Path): The filepath of the 3D model to render.
        image_width (int): The width of the output image.
        image_height (int): The height of the output image.
        radius (float): The radius parameter used to adjust the camera position.

    Returns:
        PIL.Image: The generated image as a PIL Image object.
    """
    # Load the mesh from the file
    mesh = trimesh.load(mesh_filepath)

    # Compute the centroid of the mesh
    centroid = mesh.centroid
    assert centroid.shape == (3,), f"{centroid.shape}"

    # Initialize PyVista Plotter
    pl = pv.Plotter(off_screen=True)
    pl.import_obj(mesh_filepath)

    # Set background color to white
    pl.set_background("white")

    # Compute camera position based on the radius
    camera_offset = radius / np.sqrt(3)
    camera_pos = centroid + np.array([camera_offset, camera_offset, camera_offset])
    target_pos = centroid
    up_vector = np.array([0, 1, 0])  # y-up meshes
    pl.camera_position = [
        camera_pos,  # Camera position (x, y, z)
        target_pos,  # Focal point (x, y, z)
        up_vector,  # Up vector (x, y, z)
    ]

    # Set the window size for the screenshot
    pl.window_size = [image_width, image_height]

    # Take the screenshot
    img_array = pl.screenshot(transparent_background=False)

    # Convert the screenshot (numpy array) to an image using Pillow
    img = Image.fromarray(img_array)

    # Close the plotter
    pl.close()

    return img


def main() -> None:
    """
    Main function to load the mesh, generate the image, and save it.
    """
    # Parse command-line arguments using tyro
    args = tyro.cli(MeshToImage)
    print("=" * 80)
    print(f"Parsed Arguments:\n{tyro.extras.to_yaml(args)}")
    print("=" * 80 + "\n")

    # Generate the image using the parsed arguments
    img = generate_mesh_image(
        mesh_filepath=args.mesh_filepath,
        image_width=args.image_width,
        image_height=args.image_height,
        radius=args.radius,
    )

    # Output after generating the image
    print(f"Generated image, saving to {args.output_image_path}")

    # Save the generated image
    args.output_image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output_image_path)
    print(f"Saved to {args.output_image_path}")


if __name__ == "__main__":
    main()
