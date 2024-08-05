import os
import random
import argparse
import numpy as np

#initliaze an Argument Parser
parser = argparse.ArgumentParser()

parser.add_argument('--keyframes', type=int, default=2, help='Number of keyframes to use for training set')
args = parser.parse_args()

# Directories containing the images and depth maps
images_dir = "/vol/fob-vol3/mi20/deghaisa/code/shot_0003/1_source_sequence"
depth_dir = "/vol/fob-vol3/mi20/deghaisa/code/shot_0003/2_gt_depth"

# Ensure the files are read and paired in a sorted order
image_files = sorted(os.listdir(images_dir))
depth_files = sorted(os.listdir(depth_dir))

# File to store all paired filenames
all_split_filenames = "/vol/fob-vol3/mi20/deghaisa/code/ZoeDepth/train_test_inputs/all_filenames.txt"

# Write the full paths of image and depth pairs to the combined file
with open(all_split_filenames, 'w') as file:
    for image_file, depth_file in zip(image_files, depth_files):
        # img_full_path = os.path.join(images_dir, image_file)
        img_full_path = image_file
        depth_full_path = depth_file
        file.write(f'{img_full_path} {depth_full_path} 518.8579\n')

# Define the input file and output files for splits

train_file = "/vol/fob-vol3/mi20/deghaisa/code/ZoeDepth/train_test_inputs/nyudepthv2_train_files_with_gt.txt"
# val_file = "/vol/fob-vol3/mi20/deghaisa/code/Depth-Anything-V2/metric_depth/dataset/splits/kitti/val.txt"
test_file = "/vol/fob-vol3/mi20/deghaisa/code/ZoeDepth/train_test_inputs/nyudepthv2_test_files_with_gt.txt"

train_ratio = 0.9
val_ratio = 0.1


# Read all file paths from the input file
with open(all_split_filenames, 'r') as file:
    lines = file.readlines() # a list has all the filenames

# Shuffle the list of file paths
# random.shuffle(lines)

# Compute the number of samples for each split
num_samples = len(lines)
train_end = int(train_ratio * num_samples)

selected_frames = np.zeros(num_samples, dtype=bool)

# Split the paths into train and validation sets
# train_paths = lines[:train_end]

train_paths = []
# val_paths = []
test_paths = []

keyframes = args.keyframes

# i = 0
for i in range(0, train_end, keyframes):
    train_paths.append(lines[i])
    selected_frames[i] = True
    
i +=keyframes
    
# for j in range(i, num_samples, keyframes):
#     if i < num_samples:
#         val_paths.append(lines[j])
#         selected_frames[j] = True

# for k in range(keyframes-1, num_samples, keyframes):
#     if k < num_samples:
#         test_paths.append(lines[k].split()[0]+"\n")

for k in range(0, num_samples):
    if(selected_frames[k] == 0):
        # test_paths.append(lines[k].split()[0]+"\n")
        test_paths.append(lines[k])
    
# Write the paths to the respective files
with open(train_file, 'w') as file:
    file.writelines(train_paths)

# with open(val_file, 'w') as file:
#     file.writelines(val_paths)
    
with open(test_file, 'w') as file:
    file.writelines(test_paths)

print(f"Total samples: {num_samples}")
print(f"Training samples: {len(train_paths)}")
# print(f"Validation samples: {len(val_paths)}")
print(f"Testing samples: {len(test_paths)}")