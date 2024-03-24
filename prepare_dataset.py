import os
import shutil
import random
import concurrent.futures
from pathlib import Path

def split_dataset(source_dir, train_dir, test_dir, classes, test_ratio=0.2):

    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        
        images = os.listdir(source_class_dir)
        random.shuffle(images)
        test_images = images[:int(len(images)*test_ratio)]
        train_images = images[int(len(images)*test_ratio):]
        
        for image in test_images:
            shutil.copy(os.path.join(source_class_dir, image), os.path.join(test_class_dir, image))
        
        for image in train_images:
            shutil.copy(os.path.join(source_class_dir, image), os.path.join(train_class_dir, image))

def parallel_split_dataset(source_dir, target_dirs, classes, test_ratio=0.2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for target_dir in target_dirs:
            futures.append(executor.submit(split_dataset, source_dir, target_dir, target_dirs[0], classes, test_ratio))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    source_dataset_dir = "/home/active_learning/classification/RPS_dataset"
    os.makedirs("/home/active_learning/classification/res", exist_ok=True)
    train_dir = "/home/active_learning/classification/res/split_dataset"
    test_dir = "/home/active_learning/classification/res/split_dataset"
    # valid_dir = "valid"
    classes = ["rock", "paper", "scissors"]
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # os.makedirs(valid_dir, exist_ok=True)
    
    parallel_split_dataset(source_dataset_dir, [train_dir, test_dir,], classes, test_ratio=0.2)
    
