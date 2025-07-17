import h5py
import numpy as np


import torch
import numpy as np
from skimage.transform import rotate, resize
from skimage.filters import gaussian
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode



def rotate_2d(pt_2d, rot_rad):
    """
    Rotates a 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot):
    """
    Generates the affine transformation matrix.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def generate_image_patch(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    """
    Crops an image patch and returns the patch and the transformation matrix.
    """
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    # The 'scale' parameter here is for augmentation, not the initial bbox size.
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans

# --- Helper function to transform keypoints ---

def trans_point2d(pt_2d, trans):
    """
    Transforms a 2D point using a 2x3 affine transformation matrix.
    
    Args:
        pt_2d (np.array): A 2-element array representing the (x, y) coordinate.
        trans (np.array): The 2x3 affine transformation matrix.
        
    Returns:
        np.array: The new 2-element (x, y) coordinate.
    """
    # Create a homogeneous coordinate vector [x, y, 1]
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    
    # Apply the transformation
    dst_pt = np.dot(trans, src_pt)
    
    return dst_pt















def extract_center_scale_data_from_human36m():
    # Step 1: Path to your .h5 file
    h5_file_path = 'original_data/h36m/annot/train.h5'  # Change this to your actual path

    # Step 2: Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Step 3: Print available keys to see the structure
        # print("Available keys in HDF5 file:", list(f.keys()))
        
        # Step 4: Extract 'center' and 'scale' (update keys if needed)
        center = f['center'][:]
        scale = f['scale'][:]

    # Step 5: Save to .npy files
    np.save('mmpose_data/h36m_train/center.npy', center)
    np.save('mmpose_data/h36m_train/scale.npy', scale)

    print("Saved 'center.npy' and 'scale.npy' successfully.")



import os
import numpy as np

def convert_npy_to_npz(input_folder, output_file):
    """
    Convert all .npy files in a folder to a single .npz file.

    Args:
        input_folder (str): Path to the folder containing .npy files.
        output_file (str): Path to the output .npz file.
    """
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    if not npy_files:
        print("No .npy files found in the folder.")
        return

    data = {}
    for npy_file in npy_files:
        key = os.path.splitext(npy_file)[0]  # Use the filename (without extension) as the key
        file_path = os.path.join(input_folder, npy_file)
        data[key] = np.load(file_path, allow_pickle=True)  # Enable loading object arrays
    
    np.savez(output_file, **data)
    print(f"Saved all .npy files from '{input_folder}' to '{output_file}'.")

    # Example usage
    # input_folder = 'mmpose_data/h36m_train'  # Path to the folder containing .npy files
    # output_file = 'mmpose_data/h36m_train.npz'  # Path to the output .npz file
    # convert_npy_to_npz(input_folder, output_file)