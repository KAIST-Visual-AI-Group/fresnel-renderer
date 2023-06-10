import numpy as np
from PIL import Image

from . import fresnelvis


def render_pointcloud(
    pointcloud: np.ndarray,
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0.0, 0.0, 0.0]),
    camUp=np.array([0, 1, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
    cloudR=0.006,
):
    img = fresnelvis.renderMeshCloud(
        cloud=pointcloud,
        camPos=camPos,
        camLookat=camLookat,
        camUp=camUp,
        camHeight=camHeight,
        resolution=resolution,
        samples=samples,
        cloudR=cloudR,
    )
    return Image.fromarray(img)


def render_mesh(
    vert: np.ndarray,
    face: np.ndarray,
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0, 0, 0.0]),
    camUp=np.array([0, 1, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
):
    mesh = {"vert": vert, "face": face}
    img = fresnelvis.renderMeshCloud(
        mesh=mesh,
        camPos=camPos,
        camLookat=camLookat,
        camUp=camUp,
        camHeight=camHeight,
        resolution=resolution,
        samples=samples,
    )
    return Image.fromarray(img)
