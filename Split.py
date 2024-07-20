import os
import shutil
import random

# Define the paths
base_dir = 'data/obj_train_data'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Define train, test, and valid split ratios
train_ratio = 0.7
test_ratio = 0.2
valid_ratio = 0.1

# Create directories if they don't exist
os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'valid'), exist_ok=True)

os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'valid'), exist_ok=True)

# Get a list of all image files
image_files = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]

# Shuffle the image files
random.shuffle(image_files)

# Calculate the number of files for each split
total_files = len(image_files)
train_count = int(total_files * train_ratio)
test_count = int(total_files * test_ratio)
valid_count = total_files - train_count - test_count

# Function to move files
def move_files(file_list, dest_img_dir, dest_lbl_dir):
    for file in file_list:
        base_filename = os.path.splitext(file)[0]
        img_file = f"{base_filename}.jpg"
        lbl_file = f"{base_filename}.txt"
        shutil.move(os.path.join(base_dir, img_file), os.path.join(dest_img_dir, img_file))
        shutil.move(os.path.join(base_dir, lbl_file), os.path.join(dest_lbl_dir, lbl_file))

# Split and move the files
move_files(image_files[:train_count], os.path.join(images_dir, 'train'), os.path.join(labels_dir, 'train'))
move_files(image_files[train_count:train_count + test_count], os.path.join(images_dir, 'test'), os.path.join(labels_dir, 'test'))
move_files(image_files[train_count + test_count:], os.path.join(images_dir, 'valid'), os.path.join(labels_dir, 'valid'))

print("Files have been split and moved to 'train', 'test', and 'valid' directories.")
