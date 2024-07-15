import os
import shutil

# Paths
dataset_path = 'obj_train_data'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Create directories if they don't exist
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

# Function to move files
def move_files(file_list, split):
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        image_path = os.path.join(images_path, split, file_name)
        label_path = os.path.join(labels_path, split, file_name.replace('.jpg', '.txt'))
        
        if os.path.exists(file_path):
            shutil.move(file_path, image_path)
        label_file_path = file_path.replace('.jpg', '.txt')
        if os.path.exists(label_file_path):
            shutil.move(label_file_path, label_path)

# Read file lists
with open('train.txt', 'r') as f:
    train_files = [line.strip() for line in f.readlines()]

with open('valid.txt', 'r') as f:
    valid_files = [line.strip() for line in f.readlines()]

with open('test.txt', 'r') as f:
    test_files = [line.strip() for line in f.readlines()]

# Move files
move_files(train_files, 'train')
move_files(valid_files, 'valid')
move_files(test_files, 'test')

print("Dataset split into train, valid, and test sets successfully.")
