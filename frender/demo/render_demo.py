import numpy as np
import torch

import frender.visutil as fvis

pc = np.random.normal(0, 1, 2048 * 3).reshape(2048, 3)  # samples from a unit gaussian

pc_img = fvis.render_pointcloud(pc)
pc_img.save("frender_demo_pointcloud_rendering.png")


#### render a tetrehedron ####
verts = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0, 0, 0.5],
    ]
)

faces = np.array([[2, 1, 0], [2, 0, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
mesh_img = fvis.render_mesh(verts, faces)
mesh_img.save("frender_demo_mesh_rendering.png")
