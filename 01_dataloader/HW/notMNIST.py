import os
import glob
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class NotMNIST_Dataset(Dataset):
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

notmnist_dataset = NotMNIST_Dataset(root_dir='/content/drive/MyDrive/pytorch_tutorial/pytorch_tutorial/01_dataloader/HW/notMNIST_small',
                                   transform=transforms.ToTensor())
# print(notmnist_dataset[12])
# img_array = transforms.ToPILImage()(notmnist_dataset[0][0])

dataloader = DataLoader(notmnist_dataset, batch_size=4, shuffle=True, num_workers=4)
print(next(iter(dataloader)))

print(f'total number of data : {len(notmnist_dataset)}')

for i, (images, targets) in enumerate(dataloader):
    img = images[0].numpy()
    
    plt.imshow(img.reshape((28,28)))#np.transpose(img, (1,2,0)))
    plt.show()
    # print(images.shape)
    # print(targets.shape)

    if i == 1:
        break