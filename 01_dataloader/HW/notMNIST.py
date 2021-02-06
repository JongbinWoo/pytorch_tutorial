import os
import glob
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class NotMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.annotations = self._get_annotations()
        self.label = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                "H": 7, "I": 8, "J": 9}

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1], self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        target = torch.tensor(int(self.label[self.annotations.iloc[index, 1]]))

        if self.transform:
            image = self.transform(image)
        return image, target

    def _get_annotations(self):
        df = pd.DataFrame()
        for target in os.listdir(self.root_dir):
            image_list = glob.glob(os.path.join(self.root_dir, target, '*.png'))
            df = df.append([[os.path.basename(i), os.path.dirname(i)[-1]] for i in image_list])
        return df

notmnist_dataset = NotMNISTDataset(root_dir='/content/drive/MyDrive/pytorch_tutorial/pytorch_tutorial/01_dataloader/HW/notMNIST_small',
                                   transform=transforms.ToTensor())

dataloader = DataLoader(notmnist_dataset, batch_size=4, shuffle=True, num_workers=4)
print(next(iter(dataloader)))

print(f'total number of data : {len(notmnist_dataset)}')

len_test, len_train = divmod(len(notmnist_dataset), 3)
train_set, test_set = torch.utils.data.random_split(notmnist_dataset, [len_train, len_test])

train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)

print(len(train_set))