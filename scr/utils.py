import os
import gdown
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate

from sklearn.model_selection import train_test_split


def get_data(DATA_URL, ZIP_PATH, DATA_PATH):
    gdown.download(DATA_URL, ZIP_PATH, quiet=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_PATH)
        os.remove(ZIP_PATH)


def extract_image_names(path):
    image_format = ".png"
    names = sorted([
        x for x in os.listdir(path) 
        if x.endswith(".png") or x.endswith(".jpg")
    ])
    return names


class Encoder:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.id2char = dict(enumerate(vocabulary))
        self.char2id = {v: k for k, v in self.id2char.items()}

    def encode_str(self, s):
        return torch.LongTensor([self.char2id[char] for char in s])

    def decode_str(self, torch_tensor):
        return "".join([self.id2char[id.item()] for id in torch_tensor])


def prepare_data(DATA_PATH, Encoder, test_size, val_size):
  all_images = extract_image_names(DATA_PATH)
  all_train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=33)
  train_images, val_images = train_test_split(all_train_images, test_size=val_size, random_state=33)
  vocabulary = sorted(set(list("".join([x.split('.')[0] for x in all_images]))))
  encoder = Encoder(vocabulary)
  return train_images, test_images, val_images, encoder, vocabulary


def get_dataloaders(BATCH_SIZE, train_dataset, test_dataset, val_dataset):
  train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    num_workers=2, 
    shuffle=True
  )

  val_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE,
    num_workers=2, 
    shuffle=False
  )

  test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE,
    num_workers=2, 
    shuffle=False
  )
  
  return train_loader, val_loader, test_loader


def get_test_results(predictions, test_images):
  predictions = np.concatenate(predictions)
  test_results = pd.DataFrame({
    "model_prediction": predictions, 
    "true_label": [x[:-4] for x in test_images]})
  test_results["CER"] = [CharErrorRate()(x, y).item() for x, y\
                       in zip(test_results.model_prediction, test_results.true_label)]
  
  return test_results.sort_values("CER", ascending=False)


def show_mistakes(test_results, test_dataset):
  for id, row in test_results.head(10).iterrows():
    print(f'Model prediction: {row.model_prediction}')
    print(f"True label: {row.true_label}")
    print(f"CER = {row.CER}")
    image, _, _ = test_dataset[id]
    image = image.numpy()[0]
    plt.imshow(image, cmap='gray')
    plt.show()
