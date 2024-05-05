import argparse
import os
import numpy as np
from PIL import Image
import sys

import cv2


def save_images_to_video(path, images, fps = 4):
	# Specify video parameters
	D, H, W, C = images.shape

	# Define codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can use other codecs as well, e.g., 'XVID', 'MJPG', 'DIVX'
	out = cv2.VideoWriter(path, fourcc, fps, (W, H))

	# Iterate through image paths, read each image, and write to video
	for image in images:
		out.write(image)

	# Release the VideoWriter object
	out.release()
    
# Define the path to the main folder
def main(path: str):
    main_folder = os.path.join(path, 'validation') 

    # Initialize lists to store file paths
    lhs_files = []
    rhs_files = []

    subfolder = sorted(os.listdir(main_folder), key=lambda x:int(x))
    print(subfolder)
    # Iterate through each timestamp subfolder
    for timestamp_folder in subfolder:
        timestamp_path = os.path.join(main_folder, timestamp_folder)
        
        # Check if it's a directory
        if os.path.isdir(timestamp_path):
            image_folder = os.path.join(timestamp_path, "Image")
            
            # Check if Image folder exists
            if os.path.exists(image_folder):
                lhs_folder = os.path.join(image_folder, "lhs")
                rhs_folder = os.path.join(image_folder, "rhs")
                
                # Check if lhs and rhs folders exist
                if os.path.exists(lhs_folder) and os.path.exists(rhs_folder):
                    lhs_png_folder = os.path.join(lhs_folder, "png")
                    rhs_png_folder = os.path.join(rhs_folder, "png")
                    
                    # Check if png folders exist
                    if os.path.exists(lhs_png_folder) and os.path.exists(rhs_png_folder):
                        # Check if "000.png" files exist
                        lhs_png_path = os.path.join(lhs_png_folder, "000.png")
                        rhs_png_path = os.path.join(rhs_png_folder, "000.png")
                        
                        if os.path.exists(lhs_png_path):
                            lhs_files.append(lhs_png_path)
                        
                        if os.path.exists(rhs_png_path):
                            rhs_files.append(rhs_png_path)

    # Load images into NumPy arrays
    lhs_images = np.stack([cv2.imread(filepath) for filepath in lhs_files])
    rhs_images = np.stack([cv2.imread(filepath) for filepath in rhs_files])

    print("Left-hand side images shape:", lhs_images.shape)
    print("Right-hand side images shape:", rhs_images.shape)

    D, H, W, C = lhs_images.shape
    lhs_images = np.append(np.zeros((2,H,W,C), dtype=lhs_images.dtype), lhs_images, axis=0)
    rhs_images = np.append(np.zeros((2,H,W,C), dtype=lhs_images.dtype), rhs_images, axis=0)

    save_images_to_video(os.path.join(path, 'lhs.mp4') , lhs_images)
    save_images_to_video(os.path.join(path, 'rhs.mp4') , rhs_images)

    print("Video is saved in ", path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("path", help="path to output folder. 2024-xx-xxxx", type=str)

    # Parse the arguments
    args = parser.parse_args()
    
    main(args.path)