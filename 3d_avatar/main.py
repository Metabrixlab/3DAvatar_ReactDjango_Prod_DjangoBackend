import gradio as gr
import torch
import numpy as np
import open3d as o3d
import base64
import json
import datetime
import sys
import io
import trimesh
import warnings
from PIL import Image
from pathlib import Path
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
# from azure.storage.blob import BlobServiceClient

from config import getConfig
warnings.filterwarnings('ignore')
args = getConfig()


feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

#torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

# def read_image_from_blob(blob_conn_str, blob_container_name, file_name):
#     # Connect to the Blob Storage account and get the container client
#     blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
#     container_client = blob_service_client.get_container_client(blob_container_name)
    
#     # Download the image file from the container
#     blob_client = container_client.get_blob_client(file_name)
#     image_data = blob_client.download_blob().content_as_bytes()
    
#     # Print the size of the image file
#     image_size = len(image_data)
#     print(f"Downloaded image '{file_name}' with size {image_size} bytes")
    
#     return image_data
    

def main(args):
    # blob_conn_str = "DefaultEndpointsProtocol=https;AccountName=project01rgbba3;AccountKey=9FdBLtdJWDlNvyMnG+O+R8FQ3+iUmgaiqQT4tQmkpV5XPy/mBTpB/z+w4F496qPD71ALtMx+RGlB+AStmfDWsg==;EndpointSuffix=core.windows.net"
    # blob_container_name = "api-container"
    # file_name = image_path.split("/")[-1]
    
    # print(">>>",bool(auth),">>",bool(gender))
    image_path = "./media/original/test.png"
    file_name = image_path.split("/")[-1]

    err_msg = "ALL OK!"
    image_raw = Image.open(image_path)
    image = image_raw.resize(
        (800, int(800 * image_raw.size[1] / image_raw.size[0])),
        Image.Resampling.LANCZOS
    )

    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output = prediction.cpu().numpy()
    depth_image = (output * 255 / np.max(output)).astype('uint8')

    try:
        gltf_path = create_3d_obj(np.array(image), depth_image, file_name.split(".")[0])
        # upload_file_to_blob(blob_conn_str, blob_container_name, gltf_path, file_name.split(".")[0]+".glb")
        
        # img = Image.fromarray(depth_image)
        with open(gltf_path, 'rb') as f:
            glb_data = f.read()

        encoded_data = base64.b64encode(glb_data).decode('utf-8')
        now = datetime.datetime.now()
        glb_dict = [{"glb_data" : encoded_data,
                    # "gender" : gender,
                    # "name" : auth,
                    "time" : now.strftime("D-%Y-%m-%d")+now.strftime("_T-%H-%M-%S")}]
        glb_dict1 = [{"log" : err_msg}]            
        json_string = json.dumps(glb_dict)
        json_string1 = json.dumps(glb_dict1)
        return [json_string1,json_string]


    except:
        print("Error reconstructing 3D model")
        raise Exception("Error reconstructing 3D model")

def create_3d_obj(rgb_image, depth_image, image_path, depth=10):
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(rgb_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    w = int(depth_image.shape[1])
    h = int(depth_image.shape[0])

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(w, h, 500, 500, w/2, h/2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic)

    print('normals')
    pcd.normals = o3d.utility.Vector3dVector(
        np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 1000.]))
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    pcd.transform([[-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    #print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=True)

    voxel_size = max(mesh_raw.get_max_bound() - mesh_raw.get_min_bound()) / 256
    #print(f'voxel_size = {voxel_size:e}')
    mesh = mesh_raw.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)
    gltf_path = f'./{image_path}.glb'
    o3d.io.write_triangle_mesh(
        gltf_path, mesh_crop, write_triangle_uvs=True)
    return gltf_path


title = "Demo: zero-shot depth estimation with DPT + 3D Point Cloud"
description = "This demo is a variation from the original <a href='https://huggingface.co/spaces/nielsr/dpt-depth-estimation' target='_blank'>DPT Demo</a>. It uses the DPT model to predict the depth of an image and then uses 3D Point Cloud to create a 3D object."
examples =[['cats.jpg']]

if __name__ == '__main__':
    print("!@!@!@")
    main(args)
else:
    print("&!$!5!4")