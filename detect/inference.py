import torch
import cv2
import numpy as np
from collections import OrderedDict
from .craft import CRAFT
from . import craft_utils, imgproc

# Добавьте это в начало файла
import sys
from pathlib import Path

# Добавляем путь к текущей директории для локальных импортов
sys.path.append(str(Path(__file__).parent))

# Теперь можно импортировать file_utils
try:
    import file_utils
except ImportError:
    from . import file_utils  # Альтернативный вариант импорта

def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[1:]) if k.startswith("module") else k
        new_state_dict[name] = v
    return new_state_dict

# def load_craft_model(model_path='weights/craft_mlt_25k.pth'):
#     net = CRAFT()
    
    
#     state_dict = torch.load(model_path, map_location='cpu')
#     net.load_state_dict(copyStateDict(state_dict))
#     net.eval()
#     return net



def load_craft_model(model_path=None):
    """Загрузка модели CRAFT с автоматическим определением пути"""
    # Определяем правильный путь к модели
    if model_path is None:
        current_dir = Path(__file__).parent
        model_path = current_dir / 'weights' / 'craft_mlt_25k.pth'
    
    # Проверка существования файла
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please download it from: https://github.com/clovaai/CRAFT-pytorch"
        )
    
    net = CRAFT()
    state_dict = torch.load(model_path, map_location='cpu')
    net.load_state_dict(copyStateDict(state_dict))
    net.eval()
    return net

def detect_text_region(image_path, model):
    """Обнаружение текстовых областей с улучшенной обработкой ошибок"""
    try:
        # Загрузка изображения с проверкой
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = imgproc.loadImage(image_path)
        orig = cv2.imread(image_path)
        if orig is None:
            raise ValueError("Failed to read image with OpenCV")

        # Предобработка изображения
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        ratio_h = ratio_w = 1 / target_ratio

        # Нормализация
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        # Детекция текста
        with torch.no_grad():
            y, _ = model(x)
        
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()

        # Постобработка результатов
        boxes, _ = craft_utils.getDetBoxes(
            score_text, score_link, 
            text_threshold=0.7, 
            link_threshold=0.4, 
            low_text=0.4, 
            poly=False
        )
        
        if not boxes:
            print("No text detected, returning original image")
            return image_path

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

        # Вычисление общего bounding box
        all_points = np.concatenate(boxes).reshape(-1, 2)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)

        # Добавление паддинга с проверкой границ
        pad = 10
        x_min = max(0, int(x_min) - pad)
        y_min = max(0, int(y_min) - pad)
        x_max = min(orig.shape[1], int(x_max) + pad)
        y_max = min(orig.shape[0], int(y_max) + pad)

        # Вырезание области с текстом
        cropped = orig[y_min:y_max, x_min:x_max]
        
        # Сохранение результата
        output_path = str(Path(image_path).with_stem(Path(image_path).stem + "_cropped"))
        cv2.imwrite(output_path, cropped)

        return output_path

    except Exception as e:
        print(f"Error in text detection: {str(e)}")
        return image_path  # Возвращаем оригинал при ошибке