import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
from PIL import Image

class CaptchaDataset(Dataset):
    def __init__(self, data_path, image_names, encoder, transform=None):
        self.data_path = data_path
        self.image_names = image_names 
        self.transform = transform
        self.encoder = encoder

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.data_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        image_label = self._extract_label(image_name)
        enc_label = self.encoder.encode_str(image_label)

        return image, enc_label, image_label

    @staticmethod
    def _extract_label(name):
        return name[:-4]

    def encode_label(self, label):
        return torch.LongTensor([self.char2id[char] for char in label])

    def __len__(self):
        return len(self.image_names)

def prepare_datasets(DATA_PATH, train_images, test_images, val_images, encoder):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.225),
])
  train_dataset = CaptchaDataset(DATA_PATH, train_images, encoder, transform)
  val_dataset = CaptchaDataset(DATA_PATH, val_images, encoder, transform)
  test_dataset = CaptchaDataset(DATA_PATH, test_images, encoder, transform)

  return train_dataset, val_dataset, test_dataset
