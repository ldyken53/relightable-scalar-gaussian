import numpy as np
import matplotlib.pyplot as plt

# Example parameters
num_points = 100
num_maps = 10  # example value for args.TFnums
indices = np.linspace(0, 1, num_points)
step_size = 1.0 / num_maps
eps = 1e-4

# Generate opacity maps
opacs = []
# for step in range(num_maps):
#     center = step * step_size + step_size / 2 + eps
#     arr = np.zeros(num_points, dtype=np.float32)
#     for i, x in enumerate(indices):
#         dist = abs(x - center)
#         arr[i] = min(1, max(0, 1 * (1 - (dist * 2 * 1 * (num_maps / 2)))))
#     opacs.append(arr)
# opacs = [opacs[9] + opacs[7] + opacs[5], opacs[8] + opacs[6] + opacs[4], opacs[9] + opacs[6], opacs[8] + opacs[5], opacs[4] + opacs[7]]
control_x = np.array([0.0, 0.2, 0.2, 0.3, 0.3, 1.0])
control_y = np.array([0.05, 0.05, 0.5, 0.5, 0.0, 0.0])

indices = np.linspace(0, 1, num_points)
opacs.append(np.interp(indices, control_x, control_y).astype(np.float32))

# Example selection for demonstration
cam_info = {
    "colormap": "RdYlBu",
    "opacmap": 0,  # choose one opacity map to visualize
}

# Build the colorbar representation
cmap = plt.cm.get_cmap(cam_info["colormap"])
colors = cmap(indices)  # (num_points, 4)
omap = opacs[cam_info["opacmap"]]
# omap = np.linspace(0, 1, num_points) * 2

# Apply opacity to RGBA colors
colors[:, -1] = omap

# Create the figure and render the colorbar
fig, ax = plt.subplots(figsize=(3, 0.5))
ax.imshow(
    [colors],
    extent=[0, 1, 0, 0.2],
    aspect="auto",
)
ax.set_yticks([])
ax.set_xticks(np.linspace(0, 1, 5 + 1))

# Save the colorbar as an image file (transparent background optional)
output_file = "transfer_function.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"Saved transfer function colorbar to '{output_file}'")
