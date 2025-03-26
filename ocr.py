import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model.model import TransformerModel, ALPHABET, DEVICE, WIDTH, HEIGHT, CHANNELS


MODEL_PATH = "checkpoints/AiVisionBotcheckpoint_20.pt"

# === Загрузка модели ===
def load_trained_model():
    model = TransformerModel(
        outtoken=len(ALPHABET),
        hidden=512,
        enc_layers=2,
        dec_layers=2,
        nhead=4,
        dropout=0.2
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# === Предобработка изображения ===
def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    
    # Resize to HEIGHT and keep ratio
    h, w, _ = img.shape
    new_h = HEIGHT
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    
    # Pad to WIDTH if needed
    if new_w < WIDTH:
        pad = np.full((HEIGHT, WIDTH - new_w, 3), 255, dtype=np.uint8)
        img = np.concatenate((img, pad), axis=1)
    else:
        img = cv2.resize(img, (WIDTH, HEIGHT))
    
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    tensor = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
    
    if CHANNELS == 1:
        tensor = transforms.Grayscale(CHANNELS)(tensor)

    return tensor

# === Преобразование индексов в текст ===
def indices_to_text(indexes):
    text = "".join([ALPHABET[i] for i in indexes])
    return text.replace("PAD", "").replace("SOS", "").replace("EOS", "").strip()

# === Основная функция распознавания ===
def recognize_text(image_path):
    print(f"📷 Распознаю текст на изображении: {image_path}")
    model = load_trained_model()
    tensor = process_image(image_path)
    
    with torch.no_grad():
        output_indexes = model.predict(tensor)
        prediction = indices_to_text(output_indexes[0])

    print(f"📜 Распознанный текст: {prediction}")
    return prediction

# === Пример запуска ===
if __name__ == "__main__":
    IMAGE_PATH = "test_samples/test-word5.png"  # Укажи свой путь
    recognize_text(IMAGE_PATH)