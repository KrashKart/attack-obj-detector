import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d


from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    AmbientLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    BlendParams,)

import torchvision.transforms.functional as TF
from pytorch3d.structures.meshes import join_meshes_as_scene as join_scene
from pytorch3d.renderer.camera_utils import join_cameras_as_batch as join_cameras
import torchshow as ts

from helper import *
    

def create_mesh(filepath, 
                device, 
                normalise=True, 
                position=None, 
                rotation=0.0, 
                rot_axis="Y", 
                size_scale=1):
    """Loads a mesh from an obj file and any mtl and texture file associated with the object
    
    Args:
        filepath (str): path of the .obj file to load
        device: device
        normalise (bool): whether to normalise all vertices to magnitude 1 and centered position 0
        position (float list): list of x, y, and z coordinates to position mesh at
        rotation (float): angle in degrees to rotate
        rot_axis (str): rotate along "X", "Y" or "Z" axis
        size_scale (float): scale the mesh by factor
    
    Returns:
        mesh (Meshes): mesh
    """

    mesh = load_objs_as_meshes([filepath], device=device)
    if normalise:
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))
        
    if position:
        verts_shape = mesh.verts_packed().shape
        offset = torch.full(verts_shape, 1., device=device)
        offset *= torch.tensor(position, device=device)     
        mesh.offset_verts_(offset)
    
    if size_scale != 1.0:
        mesh.scale_verts_(size_scale)
    
    if rotation != 0:
        angle = torch.tensor([rotation], device=device)
        rotator = pytorch3d.transforms.RotateAxisAngle(angle, rot_axis, degrees=True) # transforms3d class object
        new_verts_padded = rotator.transform_points(mesh.verts_padded()) # transforms padded vertex tensor
        mesh = mesh.update_padded(new_verts_padded)
        
    return mesh


def create_render(device, 
                  distance=2, 
                  elev=30, 
                  azim=30, 
                  bin_size=None, 
                  faces=None, 
                  background=[1.0, 1.0, 1.0], 
                  ambient=[1.0, 1.0, 1.0],
                  diffuse=[1.0, 1.0, 1.0],
                  specular=[1.0, 1.0, 1.0],
                  light_loc=[2.0, 2.0, 2.0],
                 ):

    """Creates a renderer with one camera
    
    Args:
        device: device
        distance (float): Distance from (0, 0, 0) to place the camera for rendering
        elev, azim (float): angle of azim and elev in degrees
        light_loc (float list): coordinates for light placement
        bin_size (int):  size of oscar's house
        faces (int): number of faces facebook has
        background (float list): RGB of desired background
    
    Returns:
        renderer: renderer
    """
    
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=360-azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    lights = PointLights(ambient_color=(ambient,), diffuse_color=(diffuse,), specular_color=(specular,), 
                         location=(light_loc,), device=device)
    blend_params = BlendParams(background_color=background)
    
    # set rasterization settings
    raster_settings = RasterizationSettings(
            image_size=640,
            blur_radius=0.0, 
            faces_per_pixel=1,
            max_faces_per_bin=faces,  # toggle max_faces_per_bin if model has too many faces
            bin_size=bin_size,
        )
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    
     # initialise renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=shader
    )
    
    return renderer

def create_cameras(device, elev_batch=10, azim_batch=10, distance=2.0, elevMin=0, elevMax=30, azimMin=0, azimMax=30):
    """Creates a list of cameras
    Args:
        device: device
        elev_batch, azim_batch (int): number of cameras to create for elev and azim each
        distance (float): Distance from (0, 0, 0) to place the cameras for rendering
        elevMin, elevMax, azimMin, azimMax (float): Max and Min angles in degrees of azim and elev
    
    Returns:
        cameras (Camera list): list of Cameras
    """
    cameras = []
    elev = torch.linspace(elevMin, elevMax, elev_batch)
    azim = torch.linspace(azimMin, azimMax, azim_batch)
    
    for e in elev:
        for a in azim:
            R, T = look_at_view_transform(dist=distance, elev=e, azim=360-a)
            camera = Camera(FoVPerspectiveCameras(device=device, R=R, T=T), distance, e, a)
            cameras.append(camera)
            
    elevInt = 0 if elev_batch == 1 else (elevMax - elevMin)/(elev_batch - 1)
    azimInt = 0 if azim_batch == 1 else (azimMax - azimMin)/(azim_batch - 1)
    print(f"{' Cameras created ':=^35}")
    print(f"{len(cameras)} cameras created")
    print(f"distance = {distance}")
    print(f"elev = {elevMin} to {elevMax} in {elevInt:.4f} degree increments")
    print(f"elev = {azimMin} to {azimMax} in {azimInt:.4f} degree increments")
    return cameras


def tt_split(cameras, train_prop):
    """Splits cameras according to train proportion
    
    Args:
        cameras (list): list of cameras
        train_prop (float): proportion of cameras to dedicate to training
        
    Returns:
        train_cams, test_cams (Camera list): FoVPerspective cameras for testing
        training_idx, test_idx (int list): index of train and test cameras w.r.t. original cameras list
    """
    
    total_views = len(cameras)
    train_size = round(train_prop * total_views)
    training_idx = np.random.choice(total_views, train_size, replace=False)
    test_idx = [idx for idx in range(total_views) if idx not in training_idx]

    print(test_idx)
    train_cams = [cameras[int(idx)] for idx in training_idx]
    test_cams = [cameras[int(idx)] for idx in test_idx]
    return train_cams, test_cams, training_idx, test_idx


def get_texture_uv(mesh):
    """Extracts the TextureUV tensor from the mesh
    
    Args:
        mesh: mesh
        
    Returns:
        texture_tensor (Tensor): Texture map [1, W, H, RGB]
    """
    
    texture_image = mesh.textures.maps_padded().clone().detach()
    return texture_image


def see_uv(mesh, **kwargs):
    """Visualise the TextureUV map of the mesh
    
    Args:
        mesh: mesh
        **kwargs: kwargs for ts.show
     
    Returns:
        None
    """
    
    uv = get_texture_uv(mesh)
    ts.show(uv, **kwargs)


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Not called directly in main.py

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

            
def render_one(mesh, renderer, device, distance=3.0, elev=45, azim=45):
    """Renders one point of view of the object
    
    Args:
        mesh: mesh
        renderer: renderer
        device: device
        distance, elev, azim (int): parameters of the camera to render on (elev and azim in degs)
        
    Returns:
        Render object rendered by the camera
    """

    # Initialise cameras 
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=360 - azim)
    cameras = Camera(FoVPerspectiveCameras(device=device, R=R, T=T), distance, elev, azim)
    
    return cameras.render(mesh, renderer)

                           
def render_batch(scene, renderer, cameras):
    """Batch rendering using cameras
    
    Args:
        renderer (renderer) : renderer
        scene (Meshes): Meshes object (unextended)
        cameras (Camera list): list of Cameras
    
    Returns:
        images (Renders list): list of Renders by the cameras
        
    """
    
    renders = []
    for camera in cameras:
        d, e, a = camera.get_params()
        image = camera.render(scene, renderer)
        renders.append(image)
    return renders


def render_batch_paste(scene, renderer, cameras,):
    """FOR PASTE ONLY. Renders scene using cameras, then finds the respective image to paste the car onto
    
    Args:
        renderer (renderer) : renderer
        scene (Meshes): Meshes object (unextended)
        cameras (cameras): cameras (not list of cameras)
    
    Returns:
        output (Render list): list of Renders
    """
    
    output = []
    for camera in cameras:
        d, e, a = camera.get_params()
        onto = search(d, e, a, torch.device("cuda:0"))
        img = camera.render(scene, renderer)
        result = img_mask(img, onto, torch.device("cuda:0"))
        output.append(Render(result, d, e, a))
    return output

                           
def render_around(mesh, renderer, device, batch_size, distance=2.0, elevMin=0, elevMax=180, azimMin=0, azimMax=360, **kwargs):
    """Used to render and visualise BATCH_SIZE number of views of the object
    
    Args:
        mesh: mesh
        renderer: renderer
        device: device
        batch_size (int): number of camera angles and subplots to render
        distance (float), elevMin, elevMax, azimMin, azimMax (int): camera parameters to render on (elev, azim in degs)
        
    Returns:
        images (Tensor): tensor of shape [batch_size, W, H, RGBA]
    """

    # initialise BATCH_SIZE number of meshes
    meshes = mesh.extend(batch_size)

    # 360 azimuth and 180 elevation camera angles split into BATCH_SIZE view
    elev = torch.linspace(elevMin, elevMax, batch_size)
    azim = 360 - torch.linspace(azimMin, azimMax, batch_size)

    # Initialise cameras 
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Put light in front of object
    lights = PointLights(device = device, location = [[0, 2.0, 2.0]])

    images = renderer(meshes, cameras=cameras, lights=lights) # [batch_size, W, H, RGBA]
    images = torch.clamp(images, min=0.0, max=1.0)
    ts.show(preprocess(images, "pil"), **kwargs)
    return images

        
def join(*args):
    """Joins a list of meshes to form a scene
    
    Args:
        *args (Meshes list): list of meshes to join to form scene
    
    Returns:
        scene (Meshes): combined scene represented by one mesh
    """
    
    if len(args) == 0:
        print("No meshes selected")
    scene = join_scene(list(args))
    return scene