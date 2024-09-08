import pyvista as pv
from PIL import Image

# Load the example textured mesh (replace 'your_mesh.obj' with the path to your mesh file)
mesh = pv.read("/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/snackbox/3DModel.obj")

# Set up plotter and scene
plotter = pv.Plotter(off_screen=True, window_size=[800, 600])

# Set the background color to white
plotter.set_background('white')

# Add the mesh to the scene
plotter.add_mesh(mesh)

# Take a screenshot
img_array = plotter.screenshot(transparent_background=False)

# Convert the screenshot (numpy array) to an image using Pillow
img = Image.fromarray(img_array)

# Save the image
img.save("textured_mesh_with_white_background.png")

plotter.close()

