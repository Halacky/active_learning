

import random
import os
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import requests

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import utils, models

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png"])

class IndexedDataset(Dataset):

    def __init__(self, dir_path, transform=None, test=False):
        '''
        Args:
        - dir_path (string): path to the directory containing images
        - transform (torchvision.transforms.) (default=None)
        - test (boolean): True for labeled images, False otherwise (default=False)
        '''

        self.dir_path = dir_path
        self.transform = transform
        
        image_filenames = []
        for (dirpath, dirnames, filenames) in os.walk(dir_path):
            image_filenames += [os.path.join(dirpath, file) for file in filenames if is_image(file)]
        self.image_filenames = image_filenames    
        
        # We assume that in the beginning, the entire dataset is unlabeled, unless it is flagged as 'test':
        if test:
            # The image's label is given by the first digit of its subdirectory's name
            # E.g. the label for the image file `./dogs/train/6_great_dane/n02109047_22481.jpg` is 6
            self.labels = [f[len(self.dir_path)+1] for f in self.image_filenames]
            self.unlabeled_mask = np.zeros(len(self.image_filenames))
        else:
            self.labels =[0]*len(self.image_filenames)
            self.unlabeled_mask = np.ones(len(self.image_filenames))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_name = self.image_filenames[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], idx
    
    # Display the image [idx] and its filename
    def display(self, idx):
        img_name = self.image_filenames[idx]
        print(img_name)
        self.send_image(img_name)

        # img=mpimg.imread(img_name)
        # imgplot = plt.imshow(img)

        # plt.show()
        return
    
    def send_image(self, image_path, class_number=0):
        url = 'http://127.0.0.1:5002/receive_image'

        with open(image_path, 'rb') as img_file:
            # Создаем словарь с изображением
            files = {'image': img_file}
            
            # Отправляем POST запрос на сервер Flask
            response = requests.post(url, files=files)

        # Выводим ответ сервера
        print(response.text)

            
    # Set the label of image [idx] to 'new_label'
    def update_label(self, idx, new_label):
        self.labels[idx] = new_label
        self.unlabeled_mask[idx] = 0
        return
    
    # Set the label of image [idx] to that read from its filename
    def label_from_filename(self, idx):
        self.labels[idx] = self.image_filenames[idx][len(self.dir_path)+1]
        self.unlabeled_mask[idx] = 0
        return
 