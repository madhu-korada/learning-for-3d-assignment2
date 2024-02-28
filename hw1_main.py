'''
@File    :   hw1_main.py
@Time    :   2024/02/08 21:46:51
@Author  :   Madhu Korada 
@Version :   1.0
@Contact :   mkorada@cs.cmu.edu
@License :   (C)Copyright 2024-2025, Madhu Korada
@Desc    :   This file contains all the tasks of the assignments.
@Usage   :   python -m main --question 1.1 
'''

import argparse
import pytorch3d
import torch
import numpy as np
import imageio
import os
import mcubes
import matplotlib.pyplot as plt

import hw1_starter.render_mesh as render_mesh
from hw1_starter.utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer, unproject_depth_image
import hw1_starter.camera_transforms 
from hw1_starter.render_generic import load_rgbd_data
from hw1_starter.dolly_zoom import dolly_zoom

def render_360_degree_view_from_mesh_file(
        mesh_path, image_size=256, output_path="images/", file_name="cow_mesh.gif", num_views=36, device=None, imshow=False
):
    """
    This function renders a 360 degree view of a mesh and saves it as a gif.
    
    @param mesh: The mesh to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    
    @return: None
    """
    images = []

    for i in range(num_views):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(2.7, 0, i * 10, degrees=True)
        image = render_mesh.render_cow_with_tf(mesh_path, image_size=512, R=R, T=T)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        images.append(image)
        
        if imshow:
            plt.imshow(image)
            plt.title('360 degree view of cow')
            plt.show(block=False)
            plt.pause(0.02)
    
    imageio.mimsave(os.path.join(output_path, file_name), images, fps=15)
    print(f"Saved render to {os.path.join(output_path, file_name)}")
    return

def render_360_degree_view_from_mesh_file(
        mesh_path, image_size=256, output_path="images/", file_name="cow_mesh.gif", num_views=36, device=None, imshow=False, dist=3
):
    """
    This function renders a 360 degree view of a mesh and saves it as a gif.
    
    @param mesh: The mesh to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    
    @return: None
    """
    images = []

    for i in range(num_views):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=0, azim=i * 10, degrees=True)
        image = render_mesh.render_cow_with_tf(mesh_path, image_size=512, R=R, T=T)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        images.append(image)
        
        if imshow:
            plt.imshow(image)
            plt.title('360 degree view of the mesh')
            plt.show(block=False)
            plt.pause(0.02)
    
    imageio.mimsave(os.path.join(output_path, file_name), images, fps=15)
    print(f"Saved render to {os.path.join(output_path, file_name)}")
    return

def render_360_degree_view_from_mesh(
        mesh, image_size=256, output_path="images/output", file_name="output.gif", num_views=36, device=None, imshow=False
):
    """
    This function renders a 360 degree view of a mesh and saves it as a gif.
    
    @param mesh: The mesh to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    
    @return: None
    """
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    renderer = get_mesh_renderer(image_size=image_size)

    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, elev=0, azim=i*10, degrees=True)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        images.append(image)
        
        if imshow:
            plt.imshow(image)
            plt.title('360 degree view of the mesh')
            plt.show(block=False)
            plt.pause(0.02)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imageio.mimsave(os.path.join(output_path, file_name), images, fps=15)
    print(f"Saved render to {os.path.join(output_path, file_name)}")
    return

def render_mesh(mesh, file_name):
    """
    This function renders a 360 degree view of a mesh and saves it as a gif.
    
    @param mesh: The mesh to render.
    @param file_name: The path to save the gif.
    
    @return: None
    """
    device=mesh.device
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0), device = device)
    textures = textures * torch.tensor([0.7, 0.7, 1], device = device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_degree_view_from_mesh(mesh.detach(), file_name=file_name, device=device)
    return None


def render_vox(vox, file_name):
    """
    This function renders a 360 degree view of a voxel and saves it as a gif.
    
    @param vox: The voxel to render.
    @param file_name: The path to save the gif.
    
    @return: None
    """
    device=vox.device
    mesh = pytorch3d.ops.cubify(vox, thresh=0.5, device=device)
    if mesh.verts_packed().shape == torch.Size([0, 3]):
        print('No mesh')
        return
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_360_degree_view_from_mesh(mesh, file_name=file_name, device=device)
    return None


def render_tetrahedron():
    """
    This function renders a 360 degree view of a tetrahedron and saves it as a gif.
    
    @return: None
    """
    # 2.1 Constructing a Tetrahedron (5 points)
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])
    # save this mesh with colors
    pytorch3d.io.save_obj("data/tetrahedron.obj", vertices, faces)

    render_360_degree_view_from_mesh_file("data/tetrahedron.obj", args.image_size, args.output_path, "tetrahedron.gif", 36, imshow=True)

def render_cube():
    """
    This function renders a 360 degree view of a cube and saves it as a gif.
    
    @return: None
    """
    # 2.2 Constructing a Cube (5 points)
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6], [0, 3, 7], [0, 4, 7], [4, 5, 6], [4, 6, 7]])
    
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])
    # save this mesh with colors
    pytorch3d.io.save_obj("data/cube.obj", vertices, faces)
    
    render_360_degree_view_from_mesh_file("data/cube.obj", args.image_size, args.output_path, "cube.gif", 36, imshow=True)


def retexture_cow_mesh(
        mesh_path, image_size=512, output_path="images/", file_name="cow_retextured.gif", num_views=36, device=None, imshow=False
):
    """
    This function retextures a cow mesh and saves it as a gif.
    
    @param mesh: The mesh to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    @param imshow: Whether to show the images as they are rendered.
    
    @return: None
    """
    renderer = get_mesh_renderer(image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(mesh_path)
    vertices, faces = vertices.unsqueeze(0), faces.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)

    textures = torch.ones_like(vertices) 
    ## Extract the z-coordinates
    z_coords = vertices[:, :, 2]
    # Find the max and min z-coordinate
    z_max, z_min = z_coords.max(), z_coords.min()

    color1, color2 = torch.tensor([1,1,1]), torch.tensor([0,1,1])
    alpha = (z_coords - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(-1)
    textures = alpha * color2 + (1 - alpha) * color1

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    render_360_degree_view_from_mesh(mesh, image_size, output_path, file_name, num_views, device=device, imshow=imshow)
    return

def camera_transforms(
        cow_path, image_size=512, output_path="images/transform_cow.jpg"
):
    """
    This function renders a cow mesh with different camera transformations.
    
    """
    # Transform 1: Rotate the cow by 90 degrees around the z-axis.
    output_path_1 = output_path.replace(".jpg", "_transform_1.jpg")
    T_1 = torch.tensor([0, 0, 0])
    R_1_euler = torch.tensor([0, 0, np.pi/2])
    R_1 = pytorch3d.transforms.euler_angles_to_matrix(R_1_euler, "XYZ").float()
    
    img_1 = starter.camera_transforms.render_cow(cow_path, image_size, R_1, T_1)
    plt.imsave(output_path_1, img_1)
    print("Saved render to ", output_path_1)
    
    # Transform 2: Translate the cow by 5 unit in the z-direction.
    output_path_2 = output_path.replace(".jpg", "_transform_2.jpg")
    T_2 = torch.tensor([0, 0, 5])
    R_2_euler = torch.tensor([0, 0, 0])
    R_2 = pytorch3d.transforms.euler_angles_to_matrix(R_2_euler, "XYZ").float()
    
    img_2 = starter.camera_transforms.render_cow(cow_path, image_size, R_2, T_2)
    plt.imsave(output_path_2, img_2)
    print("Saved render to ", output_path_2)
    
    # Transform 3: Translate the cow by 0.5 units in the x-direction.
    output_path_3 = output_path.replace(".jpg", "_transform_3.jpg")
    T_3 = torch.tensor([0.5, 0, 0])
    R_3_euler = torch.tensor([0, 0, 0])
    R_3 = pytorch3d.transforms.euler_angles_to_matrix(R_3_euler, "XYZ").float()
    
    img_3 = starter.camera_transforms.render_cow(cow_path, image_size, R_3, T_3)
    plt.imsave(output_path_3, img_3)
    print("Saved render to ", output_path_3)
    
    # Transform 4: Rotate the cow by 90 degrees around the y-axis and then translate the cow by 3 unit in the x, z direction.
    output_path_4 = output_path.replace(".jpg", "_transform_4.jpg")
    T_4 = torch.tensor([3, 0, 3])
    R_4_euler = torch.tensor([0, -np.pi/2, 0])
    R_4 = pytorch3d.transforms.euler_angles_to_matrix(R_4_euler, "XYZ").float()
    
    img_4 = starter.camera_transforms.render_cow(cow_path, image_size, R_4, T_4)
    plt.imsave(output_path_4, img_4)
    print("Saved render to ", output_path_4)
    return

def render_360_degree_view_from_pcd(
        pcd, image_size=256, output_path="images/output", file_name="output.gif", num_views=36, device=None, imshow=False
):
    """
    This function renders a 360 degree view of a mesh and saves it as a gif.
    
    @param pcd: The pcd to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    
    @return: None
    """
    
    renderer = get_points_renderer(image_size=image_size, device=device, background_color=(1, 1, 1))
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, elev=0, azim=i*10, degrees=True)
        R_fix = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float() #Flip upside down
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R @ R_fix, T=T, fov=60, device=device)


        rend = renderer(pcd, cameras=cameras)
        image = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        images.append(image)
        
        if imshow:
            plt.imshow(image)
            plt.title('360 degree view of the PCD')
            plt.show(block=False)
            plt.pause(0.02)
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imageio.mimsave(os.path.join(output_path, file_name), images, fps=15)
    print(f"Saved render to {os.path.join(output_path, file_name)}")
    return

def render_point_clouds(points, file_name):
    """
    This function renders a point cloud and saves it as a gif.
    
    @param points: The point cloud to render.
    @param file_name: The path to save the gif.
    
    @return: None
    """
    device=points.device
    points = points.reshape(1, -1, 3)
    points = points.detach()[0]
    color = (points - points.min()) / (points.max() - points.min())
    pc = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    render_360_degree_view_from_pcd(pc, file_name=file_name, device=device)
    return
    
def render_pcds_from_rgbd(
        rgbd_file="data/rgbd_data.pkl", image_size=512, output_path="images/", file_name="rgbd_pcd.gif", num_views=36, device=None, imshow=False
):
    """
    This function renders point clouds from RGB-D images and saves it as a gif.
    
    @param data: The data to render.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    @param imshow: Whether to show the images as they are rendered.
    
    @return: None
    """
    
    data = load_rgbd_data(rgbd_file)
    
    pcd_1, rgb_1 = unproject_depth_image(torch.tensor(data['rgb1']), 
                                               torch.tensor(data['mask1']), 
                                               torch.tensor(data['depth1']), 
                                               data['cameras1'])
    pcd_1, rgb_1 = pcd_1.unsqueeze(0), rgb_1.unsqueeze(0)
    torch_pcd_1 = pytorch3d.structures.Pointclouds(points=pcd_1, features=rgb_1).to(device)

    pcd_2, rgb_2 = unproject_depth_image(torch.tensor(data['rgb2']), 
                                               torch.tensor(data['mask2']), 
                                               torch.tensor(data['depth2']),
                                               data['cameras2'])
    pcd_2, rgb_2 = pcd_2.unsqueeze(0), rgb_2.unsqueeze(0)
    torch_pcd_2 = pytorch3d.structures.Pointclouds(points=pcd_2, features=rgb_2).to(device)

    torch_pcd_combined = pytorch3d.structures.join_pointclouds_as_scene([torch_pcd_1, torch_pcd_2])

    file_name_1 = file_name.replace(".gif", "_1.gif")
    render_360_degree_view_from_pcd(torch_pcd_1, image_size, output_path, file_name_1, num_views, device=device, imshow=imshow)

    file_name_2 = file_name.replace(".gif", "_2.gif")
    render_360_degree_view_from_pcd(torch_pcd_2, image_size, output_path, file_name_2, num_views, device=device, imshow=imshow)
    
    file_name_combined = file_name.replace(".gif", "_combined.gif")
    render_360_degree_view_from_pcd(torch_pcd_combined, image_size, output_path, file_name_combined, num_views, device=device, imshow=imshow)
    return

def render_parametric_function(
        num_samples=100, image_size=512, output_path="images/", file_name="parametric.gif", num_views=36, device=None, imshow=False
    ):
    """
    This function renders a 360 degree view of a torus and a hyperboloid and saves it as a gif.
    
    @param num_samples: The number of samples to use for the parametric function.
    @param image_size: The size of the images.
    @param output_path: The path to save the gif.
    @param file_name: The name of the gif.
    @param num_views: The number of views to render.
    @param device: The device to render on.
    @param imshow: Whether to show the images as they are rendered.
    
    @return: None
    """
    # Parameters for the torus
    R, r = 1, 0.5 # Major and minor radius
    
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # create a grid of phi and theta values
    U, V = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(V)) * torch.cos(U)
    y = (R + r * torch.cos(V)) * torch.sin(U)
    z = r * torch.sin(V)
    
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torch_pcd_torous = pytorch3d.structures.Pointclouds(points=[points], features=[color],).to(device)
    file_name_torus = file_name.replace(".gif", "_torous.gif")
    render_360_degree_view_from_pcd(torch_pcd_torous, image_size, output_path, file_name_torus, num_views, device=device, imshow=imshow)

    # Parameters for the hyperboloid
    a = b = 0.5  # Radii in the x and y directions
    c = 0.2      # Radius in the z direction

    # Create a grid of values for theta and z using PyTorch
    theta = torch.linspace(0, 2 * np.pi, 100)
    z = torch.linspace(-2, 2, 100)
    Theta, Z = torch.meshgrid(theta, z)

    # Parametric equations for the hyperboloid using PyTorch
    X = a * torch.cosh(Z) * torch.cos(Theta)
    Y = b * torch.cosh(Z) * torch.sin(Theta)

    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    
    torch_pcd_hyperboloid = pytorch3d.structures.Pointclouds(points=[points], features=[color],).to(device)
    file_name_hyperboloid = file_name.replace(".gif", "_hyperboloid.gif")
    render_360_degree_view_from_pcd(torch_pcd_hyperboloid, image_size, output_path, file_name_hyperboloid, num_views, device=device, imshow=imshow)
    return
    

def render_implicit_functions(
        image_size=512, output_path="images/", file_name="implicit.gif", num_views=36, device=None, imshow=False
):
    """
    This function renders a 360 degree view of a torus and saves it as a gif.
    
    """
    voxel_size = 50
    max_bound, min_bound = 1.1, -1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_bound, max_bound, voxel_size)] * 3)

    # Parameters for the torus
    R, r = 0.5, 0.25 # Major and minor radius 
    
    # Torus Equation: (R - sqrt(x^2 + y^2))^2 + z^2 - r^2 = 0
    voxels = (R - torch.sqrt(X**2 + Y**2) )**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Normalize the vertices to be in the range [-1, 1]
    vertices = (vertices / voxel_size) * (max_bound - min_bound) + min_bound
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    file_name_torus = file_name.replace(".gif", "_torus.gif")
    render_360_degree_view_from_mesh(mesh, image_size, output_path, file_name_torus, num_views, device=device, imshow=imshow)

    # Parameters for the Hyperboloid
    voxel_size = 100
    min_value = -2
    max_value = 2
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    # Parameters for the hyperboloid (a=b=c=1 for simplicity)
    a = b = c = 1

    # Implicit equation for a one-sheet hyperboloid. Equation: x^2 + y^2 - z^2 - 1 = 0
    voxels = (X**2 / a**2) + (Y**2 / b**2) - (Z**2 / c**2) - 1

    # Use marching cubes to extract the mesh
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels.numpy()), 0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Normalize the vertex coordinates
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    # Prepare textures for rendering (simple normalization here for demonstration)
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    # Create the mesh object
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    file_name_hyperboloid = file_name.replace(".gif", "_hyperboloid.gif")
    render_360_degree_view_from_mesh(mesh, image_size, output_path, file_name_hyperboloid, num_views, device=device, imshow=imshow)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="6")
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--cow_axis_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--rgbd_file", type=str, default="data/rgbd_data.pkl")
    parser.add_argument("--output_path", type=str, default="images/")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imshow", type=bool, default=False)
    parser.add_argument("--num_views", type=int, default=36)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--gif_file_name", type=str, default="cow360.gif")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    device = get_device()
    # question = "6"
    
    match args.question:
        case "1.1":
            # 1.1. 360-degree Renders (5 points)
            render_360_degree_view_from_mesh_file(
                mesh_path=args.cow_path, 
                image_size=args.image_size, 
                output_path=args.output_path, 
                file_name="cow360.gif", 
                num_views=36, 
                imshow=args.imshow
            )
        case "1.2":
            # 1.2. Re-creating the Dolly Zoom (10 points)
            dolly_zoom(
                image_size=args.image_size,
                num_frames=args.num_frames,
                duration=args.duration,
                output_file=args.gif_file_name.replace("cow", "images/dolly"),
            )
        case "2.1":
            # 2.1 Constructing a Tetrahedron (5 points)
            render_tetrahedron() 
        case "2.2":
            # 2.2 Constructing a Cube (5 points)
            render_cube()
        case "3":
            # 3. Re-texturing a mesh (10 points)
            retexture_cow_mesh(
                mesh_path=args.cow_path, 
                image_size=args.image_size, 
                output_path=args.output_path, 
                file_name="cow_retextured.gif", 
                num_views=36, 
                device=device, 
                imshow=args.imshow
            )
        case "4":
            # 4. Camera Transformations (20 points)
            camera_transforms(args.cow_axis_path, args.image_size)
        case "5.1":
            # 5.1 Rendering Point Clouds from RGB-D Images (10 points)
            render_pcds_from_rgbd(
                rgbd_file=args.rgbd_file, 
                image_size=args.image_size,
                output_path=args.output_path,
                file_name="rgbd_pcd.gif",
                num_views=args.num_views,
                device=device,
                imshow=args.imshow
            )
        case "5.2":
            # 5.2 Parametric Functions (10 + 5 points)
            render_parametric_function(
                num_samples=100, 
                image_size=512, 
                output_path="images/", 
                file_name="parametric.gif", 
                num_views=36, 
                device=device, 
                imshow=args.imshow
            )
        case "5.3":
            # 5.3 Implicit Functions (10 points)
            render_implicit_functions(
                image_size=args.image_size, 
                output_path=args.output_path, 
                file_name="implicit.gif", 
                num_views=36, 
                device=device,
                imshow=args.imshow
            )
        case "6":
            # 6. Do Something Fun (10 points)
            render_360_degree_view_from_mesh_file(
                mesh_path="data/minion.obj", 
                image_size=args.image_size, 
                output_path=args.output_path, 
                file_name=args.gif_file_name.replace("cow", "minion"),
                num_views=args.num_views, 
                imshow=args.imshow, 
                dist=100
            )
        
        case _:
            print("Invalid question number")
            # break
        
    print("Done")
    
    
    
    
