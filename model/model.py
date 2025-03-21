import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Пути к данным
DATASET_PATH = "data/cyrillic_handwriting/"
TRAIN_TSV = os.path.join(DATASET_PATH, "train.tsv")
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")

# 🔤 Создаём алфавит (уникальные символы в датасете)
train_tsv = pd.read_csv(TRAIN_TSV, sep="\t", header=None, names=["filename", "text"])
all_texts = train_tsv["text"].dropna().astype(str).tolist()  # Пропускаем NaN и превращаем всё в строки
unique_chars = sorted(set("".join(all_texts)))  # Уникальные символы

# Маппинг "символ → индекс"
char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}
char_to_index["<PAD>"] = 0  # Добавляем паддинг

index_to_char = {idx: char for char, idx in char_to_index.items()}

# Функция токенизации текста (буква → индекс)
def text_to_indices(text, max_length=32):
    text = str(text)  # Приводим к строке
    indices = [char_to_index[char] for char in text if char in char_to_index]
    indices += [0] * (max_length - len(indices))  # Паддинг
    return indices[:max_length]

# Функция обработки изображения
def process_image(image_path, target_size=(128, 32)):
    img = Image.open(image_path).convert("L")  # Ч/б изображение
    img = img.resize(target_size)  # Масштабируем
    img_array = np.array(img) / 255.0  # Нормализация (0-1)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # (1, 128, 32)
    return img_tensor

# PyTorch Dataset
class CyrillicDataset(Dataset):
    def __init__(self, tsv_path, images_path, max_text_length=32):
        self.data = pd.read_csv(tsv_path, sep="\t", header=None, names=["filename", "text"])
        self.images_path = images_path
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["filename"]
        img_path = os.path.join(self.images_path, img_name)

        # Загружаем и обрабатываем изображение
        image = process_image(img_path)

        # Токенизируем текст
        text = self.data.iloc[idx]["text"]
        text_indices = text_to_indices(text, max_length=self.max_text_length)
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        return image, text_tensor

# Тестируем Dataset
dataset = CyrillicDataset(TRAIN_TSV, TRAIN_IMAGES_PATH)

# Загружаем 1-е изображение и текст
sample_image, sample_text_tensor = dataset[0]

# Создаём DataLoader для обучения
BATCH_SIZE = 16  # Размер пакета данных

# DataLoader загружает данные пакетами
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Тестируем: Загружаем 1 batch
batch_images, batch_texts = next(iter(train_loader))







class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN: Извлекаем признаки
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))  # Сохраняем ширину
        )

        # Уменьшаем размерность перед LSTM
        self.fc1 = nn.Linear(1024, 256)  # Преобразуем 1024 → 256

        # LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)

        # Финальный слой предсказания символов
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, width, 1024]
        x = self.fc1(x)  # Преобразуем 1024 → 256
        x, _ = self.lstm(x)  # LSTM обработка
        x = self.fc2(x)  # Финальный слой
        return x


# Определяем количество классов (символов + паддинг)
num_classes = len(char_to_index)

# Создаём модель
model = CRNN(num_classes)

# Проверяем вывод
sample_output = model(batch_images)






# 🔥 Добавляем CTC Loss и код обучения

# Функция потерь CTC
ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)  # `blank=0` → паддинг

# Оптимизатор (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Функция для обучения
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Режим обучения
    epoch_loss = 0

    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)

        # Предсказание модели
        outputs = model(images)  # [batch, width, num_classes]

        # Создаём входные и выходные размеры для CTC Loss
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
        target_lengths = torch.sum(texts != 0, dim=1)  # Длина каждого текста (без паддинга)

        # Переставляем оси: CTC ждёт [width, batch, num_classes]
        outputs = outputs.permute(1, 0, 2)

        # Вычисляем loss
        loss = criterion(outputs.log_softmax(2), texts, input_lengths, target_lengths)
        epoch_loss += loss.item()

        # Оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(dataloader)

# Переносим модель на GPU (если есть)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔥 Запуск одной эпохи обучения
epoch_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)



# 🔥 Код для сохранения и загрузки модели

# Функция сохранения модели
def save_model(model, epoch, loss, path="model_checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"✅ Модель сохранена после {epoch} эпох с loss={loss:.4f}")

# Функция загрузки модели
def load_model(model, path="model_checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Загружена модель после {checkpoint['epoch']} эпох с loss={checkpoint['loss']:.4f}")
        return model, checkpoint["epoch"], checkpoint["loss"]
    else:
        print("❌ Файл модели не найден! Начинаем обучение с нуля.")
        return model, 0, None







def ctc_decode(predictions):
    pred_indices = torch.argmax(predictions, dim=2)
    decoded_texts = []
    for pred in pred_indices:
        decoded = []
        prev = None
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev:
                decoded.append(index_to_char.get(idx, ""))
            prev = idx
        decoded_texts.append("".join(decoded))
    return decoded_texts



def validate_batch(model, dataloader, index_to_char, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(dataloader))
        images, targets = images.to(device), targets.to(device)

        output = model(images)  # [B, W, C]
        output = output.cpu()  # в CPU для декодера
        decoded = ctc_decode(output)

        print("\n📊 Валидация на батче:")
        for i in range(min(num_samples, len(decoded))):
            true_indices = targets[i].tolist()
            true_text = "".join([index_to_char.get(idx, "") for idx in true_indices if idx != 0])
            print(f"🔹 GT  : {true_text}")
            print(f"🔸 Pred: {decoded[i]}")
            print("---")


# 🔥 Запускаем полное обучение
NUM_EPOCHS = 50  # Количество эпох
SAVE_EVERY = 5  # Сохраняем модель каждые 5 эпох

# Загружаем модель (если уже есть сохранение)
model, start_epoch, _ = load_model(model)

for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
    epoch_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
    print(f"🔥 Эпоха {epoch}/{NUM_EPOCHS}, loss: {epoch_loss:.4f}")

    # Сохраняем модель каждые 5 эпох
    if epoch % SAVE_EVERY == 0:
        save_model(model, epoch, epoch_loss)
        validate_batch(model, train_loader, index_to_char, device)

print("✅ Обучение завершено!")





