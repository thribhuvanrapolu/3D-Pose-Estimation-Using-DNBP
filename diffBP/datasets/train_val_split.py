import numpy as np
import os
import random
from collections import defaultdict
import json

def train_val_split_h36m(npz_file_path, 
                         split=0.8, 
                         train_frame_chunk_size=50, 
                         val_frame_chunk_size=100,
                         print_stats=True):
    """
    Splits the Human3.6M dataset into training and validation sets based on actions.
    
    Args:
        imagefiles_npy_path (str): Path to the numpy file containing image paths.
        split (float): Proportion of data to use for training (default is 0.8).
        train_frame_chunk_size (int): Number of frames per chunk for training (default is 50).
        val_frame_chunk_size (int): Number of frames per chunk for validation (default is 100).
        
    Returns:
        train_dict (dict): Dictionary containing training data, where keys are action names
            with "_partX" suffix and values are lists of (image_path, original_index)
            tuples.
        val_dict (dict): Dictionary containing validation data, structured similarly to
            train_dict.
    """
    
    all_data = np.load(npz_file_path, allow_pickle=True)
    
    # Load image names from the numpy file
    img_names = all_data['image_path']

    # --- Step 1: Group all images by action, storing both path and original index ---
    initial_image_dict = defaultdict(list)  
    # We iterate with an index so we can store it along with the path.
    initial_image_dict = defaultdict(list)
    for i, path in enumerate(img_names):
        directory_path = os.path.dirname(path)
        key = os.path.basename(directory_path)
        # Append a tuple containing the path and its original index
        initial_image_dict[key].append((path, i))

    # --- Step 2: Create a shuffled list of actions for splitting ---
    # Get all unique action keys and shuffle them randomly.
    # This ensures that our train/test split is not biased.
    all_actions = list(initial_image_dict.keys())
    random.shuffle(all_actions)

    # --- Step 3: Split actions into training (80%) and testing (20%) sets ---
    split_index = int(len(all_actions) * split)
    train_actions = all_actions[:split_index]
    val_actions = all_actions[split_index:]

    # --- Step 4: Process actions to create the final train and val dictionaries ---

    def process_and_chunk_actions(action_list, frame_chunk_size):
        """
        Takes a list of actions and a chunk size, then splits each action's
        frames into chunks of that size. The frames are expected to be
        (path, index) tuples.
        """
        final_dict = {}
        for action_key in action_list:
            # The image_list now contains (path, index) tuples
            image_list = initial_image_dict[action_key]
            
            # Calculate how many full parts can be made.
            num_parts = len(image_list) // frame_chunk_size
            
            for i in range(num_parts):
                # Create the new key with the "_partX" suffix.
                part_key = f"{action_key}_part{i + 1}"
                
                # Calculate the start and end index for the slice.
                start_index = i * frame_chunk_size
                end_index = start_index + frame_chunk_size
                
                # Slice the list to get the chunk of (path, index) tuples.
                frame_chunk = image_list[start_index:end_index]
                
                # Add the chunk to our final dictionary.
                final_dict[part_key] = frame_chunk
                
        return final_dict

    # Create the training dictionary with chunks of 50 frames
    train_dict = process_and_chunk_actions(train_actions, frame_chunk_size=train_frame_chunk_size)

    # Create the testing dictionary with chunks of 100 frames
    val_dict = process_and_chunk_actions(val_actions, frame_chunk_size=val_frame_chunk_size)

    if print_stats:

        print(f"Total actions: {len(all_actions)}")
        print(f"Training actions: {len(train_actions)}")
        print(f"Validation actions: {len(val_actions)}")
        print("-" * 20)
        
        # --- Final Output ---
        print(f"Total keys in train_dict: {len(train_dict)}")
        print(f"Total keys in val_dict: {len(val_dict)}")
        print("\nExample train_dict entry:")
        if train_dict:
            first_train_key = list(train_dict.keys())[0]
            print(f"  Key: '{first_train_key}'")
            print(f"  Value (first item): {train_dict[first_train_key][0]}")
            print(f"  Frame count: {len(train_dict[first_train_key])}")

        print("\nExample val_dict entry:")
        if val_dict:
            first_val_key = list(val_dict.keys())[0]
            print(f"  Key: '{first_val_key}'")
            print(f"  Value (first item): {val_dict[first_val_key][0]}")
            print(f"  Frame count: {len(val_dict[first_val_key])}")


    return train_dict, val_dict

