import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import utils
import matplotlib.pyplot as plt

class PoseSequenceDataset(Dataset):
    """
    Custom dataset for Human3.6M pose sequences.
    This class is structured to return a sequence of frames (a video clip)
    and their corresponding keypoint annotations for each item.
    """
    def __init__(self, data_dict, npz_file_path, img_dir, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], BBOX_SHAPE=[128, 128]):
        """
        Initializes the dataset.

        Args:
            data_dict (dict): The dictionary (e.g., train_dict or test_dict) mapping
                              a sequence key to a list of (image_path, original_index) tuples.
            npz_file_path (str): Path to the original .npz file containing all annotations.
            img_dir (str): Directory where the actual image files are stored.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            BBOX_SHAPE (list): Desired output patch shape [H, W] for cropping
        """
        self.data_dict = data_dict
        self.img_dir = img_dir
        
        # Create a list of sequence keys to easily index into the dictionary
        self.sequence_keys = list(self.data_dict.keys())

        # --- Load ALL annotations into memory once ---
        # This is more efficient than loading from the file for every item.
        print(f"Loading all annotations from {npz_file_path} into memory...")
        all_data = np.load(npz_file_path, allow_pickle=True)
        self.all_centers = all_data['center']
        self.all_scales = all_data['scale']
        self.all_keypoints_2d = all_data['keypoints2d']
        self.all_keypoints_3d = all_data['keypoints3d']
        print("Annotations loaded successfully.")

        # --- Define constants for processing ---
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.BBOX_SHAPE = BBOX_SHAPE # Desired output patch shape (H, W)

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.sequence_keys)

    def __getitem__(self, idx):
        """
        Retrieves a full sequence (video clip) and its annotations.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict: A dictionary containing the sequence of image patches and
                  the corresponding sequence of 2D keypoints.
        """
        # --- 1. Get the data for the requested sequence ---
        sequence_key = self.sequence_keys[idx]
        sequence_info = self.data_dict[sequence_key] # List of (path, original_index)

        image_patch_sequence = []
        keypoints_2d_sequence = []

        # --- 2. Process each frame in the sequence ---
        for img_path, original_index in sequence_info:
            
            # --- a. Load annotations for this specific frame ---
            center = self.all_centers[original_index].copy()
            scale = self.all_scales[original_index].copy()
            keypoints_2d_frame = self.all_keypoints_2d[original_index]
            bbox_size = scale * 200

            # --- b. Load and process the image ---            
            # FIX: Extract only the filename from the path stored in the .npz file
            filename_only = os.path.basename(img_path)
            full_image_path = os.path.join(self.img_dir, filename_only)

            full_image_bgr = cv2.imread(full_image_path)
            if full_image_bgr is None:
                print(f"Warning: Could not read image at {full_image_path}. Skipping frame.")
                continue
            full_image_rgb = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2RGB)

            # --- c. Generate cropped image patch and transformation matrix ---
            img_patch_rgb, trans = utils.generate_image_patch(
                full_image_rgb,
                center[0], center[1],
                bbox_size, bbox_size,
                self.BBOX_SHAPE[1], self.BBOX_SHAPE[0], # (W, H) for generate_patch
                do_flip=False, scale=1.0, rot=0.0
            )

            # --- d. Normalize the image patch ---
            normalized_patch = (img_patch_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
            
            # Transpose from (H, W, C) to (C, H, W) for PyTorch
            normalized_patch = normalized_patch.transpose((2, 0, 1))
            image_patch_sequence.append(normalized_patch)

            # --- e. Transform 2D keypoints to the patch coordinate system ---
            # We process all keypoints to maintain a consistent shape for stacking.
            num_keypoints = keypoints_2d_frame.shape[0]
            transformed_kps = np.zeros((num_keypoints, 2), dtype=np.float32)
            
            for i in range(num_keypoints):
                # We only transform visible keypoints
                if keypoints_2d_frame[i, 2] > 0:
                    transformed_kps[i] = utils.trans_point2d(keypoints_2d_frame[i, :2], trans)
            
            keypoints_2d_sequence.append(transformed_kps)

        # --- 3. Stack the frames and keypoints into sequence tensors ---
        if not image_patch_sequence:
            # Handle case where a sequence had no valid images
            return self.__getitem__((idx + 1) % len(self))

        # Stack list of (C, H, W) arrays into a (T, C, H, W) tensor
        video_tensor = torch.from_numpy(np.stack(image_patch_sequence, axis=0))
        
        # Stack list of (Num_KP, 2) arrays into a (T, Num_KP, 2) tensor
        keypoints_tensor = torch.from_numpy(np.stack(keypoints_2d_sequence, axis=0))
        
        # Normalize keypoints from pixel space to [-1, 1] range
        keypoints_tensor[..., 0] = (keypoints_tensor[..., 0] / self.BBOX_SHAPE[0]) * 2 - 1 # normalize x by width
        keypoints_tensor[..., 1] = (keypoints_tensor[..., 1] / self.BBOX_SHAPE[1]) * 2 - 1 # normalize y by height


        item = {
            'window': video_tensor,
            'labels': keypoints_tensor
        }

        return item
    
     
    def visualize_data(self,full_image_rgb, img_patch_rgb, transformed_keypoints, original_keypoints, center, bbox_size):
        """
        Visualizes the pre-processing steps for a single sample.
        This function is independent of the dataset class.
        
        Args:
            full_image_rgb (np.array): The original, full-size image.
            img_patch_rgb (np.array): The un-normalized cropped image patch.
            transformed_keypoints (np.array): Keypoints in the patch coordinate system.
            original_keypoints (np.array): Keypoints in the original image coordinate system.
            center (np.array): The center of the bounding box.
            bbox_size (float): The size (width/height) of the bounding box in pixels.
        """
        # Draw keypoints on the patch for visualization
        vis_img_patch = img_patch_rgb.copy()
        for kp in transformed_keypoints:
            p_x, p_y = int(round(kp[0])), int(round(kp[1]))
            cv2.circle(vis_img_patch, (p_x, p_y), radius=2, color=(0, 255, 0), thickness=-1)

        # --- Display the results ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Data Visualization", fontsize=16)

        # Panel 1: Original Image
        axes[0].imshow(full_image_rgb)
        axes[0].set_title('Original Image')
        for kp in original_keypoints:
            if kp[2] > 0:
                axes[0].scatter(kp[0], kp[1], s=25, c='lime', marker='o')
        rect = plt.Rectangle(
            (center[0] - bbox_size / 2, center[1] - bbox_size / 2),
            bbox_size, bbox_size, linewidth=2, edgecolor='yellow', facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].axis('off')

        # Panel 2: Cropped Patch
        axes[1].imshow(vis_img_patch)
        axes[1].set_title('Cropped Patch with Transformed Keypoints')
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('data_visualization.png')
        
