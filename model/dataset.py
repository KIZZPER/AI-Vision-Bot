import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# Пути к файлам EMNIST Balanced
TRAIN_CSV = "data/emnist/emnist-balanced-train.csv"
TEST_CSV = "data/emnist/emnist-balanced-test.csv"
MAPPING_TXT = "data/emnist/emnist-balanced-mapping.txt"

def load_mapping(mapping_path):
    """Загружает соответствие классов символам из файла"""
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            key, value = line.strip().split()
            mapping[int(key)] = chr(int(value))
    return mapping

class EMNISTDataset(Dataset):
    def __init__(self, csv_path, mapping, transform=None):
        self.data = pd.read_csv(csv_path, header=None).values
        self.images = self.data[:, 1:].reshape(-1, 28, 28).astype(np.uint8)  # Преобразуем пиксели в 28x28
        self.labels = self.data[:, 0].astype(int)  # Метки классов
        self.mapping = mapping
        self.transform = transform

        self.idx_to_char = {idx: char for idx, char in enumerate(sorted(set(mapping.values())))}
        self.char_to_idx = {char: idx for idx, char in self.idx_to_char.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        char = self.mapping[label]  # Символ, соответствующий классу
        label_idx = self.char_to_idx[char]  # Переводим в индекс

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # Нормализация [0,1]
        img = torch.rot90(img, k=3, dims=[1, 2])  # Поворот
        img = torch.flip(img, dims=[2])  # Отражение по горизонтали

        if self.transform:
            img = self.transform(img.numpy())  # Передаём в виде numpy для ToTensor()

        return img, label_idx
