import torch
import cv2
import os
import io
import csv
import editdistance
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model.model import TransformerModel, ALPHABET, DEVICE, WIDTH, HEIGHT, CHANNELS


# Пути
MODEL_PATH = "checkpoints/AiVisionBotcheckpoint_20.pt"
DICTIONARY_PATH = 'data/russian.txt'
FREQ_DICT_PATH = "data/freqrnc2011.csv"


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


# === Загрузка словаря ===
def load_frequency_dictionary(path=FREQ_DICT_PATH):
    freq_dict = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  
        for row in reader:
            lemma, pos, freq, r, d, doc = row
            if float(freq) > 10 and pos in ['s', 'a', 'v', 'adv']:  
                freq_dict[lemma] = float(freq)
    return freq_dict


# === Загрузка резервного словаря ===
def load_dictionary(path=DICTIONARY_PATH):
    with open(path, encoding='windows-1251') as f:
        return [line.strip() for line in f if line.strip()]


# === Предобработка изображения ===
def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    
    h, w, _ = img.shape
    new_h = HEIGHT
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    
    if new_w < WIDTH:
        pad = np.full((HEIGHT, WIDTH - new_w, 3), 255, dtype=np.uint8)
        img = np.concatenate((img, pad), axis=1)
    else:
        img = cv2.resize(img, (WIDTH, HEIGHT))
    
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))  
    tensor = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
    
    if CHANNELS == 1:
        tensor = transforms.Grayscale(CHANNELS)(tensor)

    return tensor

# === Преобразования изображения для правильного обнаружения строк ===
def preprocess_for_line_detection(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    no_horizontal = cv2.subtract(inv, detected_lines)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(no_horizontal, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cleaned = cv2.subtract(no_horizontal, detected_lines)

    cleaned = cv2.bitwise_not(cleaned)
    return cleaned


# === Преобразование индексов в текст ===
def indices_to_text(indexes):
    text = "".join([ALPHABET[i] for i in indexes])
    return text.replace("PAD", "").replace("SOS", "").replace("EOS", "").strip()


# === Исправление текста через словарь расстоянием Левенштейна ===
def correct_word(word, dictionary, freq_dict, max_dist=2):
    if not dictionary or not word:
        return word
    
    if len(word) == 1:
        return word  

    match = re.match(r"(^[^а-яА-ЯёЁ]*)([а-яА-ЯёЁ\-]+)([^а-яА-ЯёЁ]*$)", word)
    if not match:
        return word  

    prefix, core, suffix = match.groups()
    core_lower = core.lower()

    if core_lower in freq_dict:
        corrected = core_lower if core[0].islower() else core_lower.capitalize()
        return prefix + corrected + suffix

    min_len = max(1, len(core) - 2)
    max_len = len(core) + 2
    filtered_dict = [w for w in dictionary if min_len <= len(w) <= max_len]

    if not filtered_dict:
        return word

    distances = [
        (w, editdistance.eval(core_lower, w.lower()), freq_dict.get(w.lower(), 0))
        for w in filtered_dict
    ]
    distances.sort(key=lambda x: (x[1], -x[2]))  

    best_match, dist, _ = distances[0]

    if dist > max_dist:
        return word

    corrected = best_match.capitalize() if core[0].isupper() else best_match
    return prefix + corrected + suffix

# === Нахождение слов в строке ===
def segment_sentence(image_original, image_for_detection, min_word_width=25, min_word_height=20,
                     padding=20, debug=True, save_dir="cropped_words"):
    if image_original is None or image_for_detection is None:
        print("[ERROR] Одно из изображений не загружено.")
        return []

    gray = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2GRAY) if len(image_for_detection.shape) == 3 else image_for_detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))

    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    word_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_word_width and h > min_word_height:
            word_boxes.append((x, y, w, h))

    word_boxes.sort(key=lambda box: box[0])

    word_images = []
    if save_dir and debug:
        os.makedirs(save_dir, exist_ok=True)

    img_h, img_w = image_original.shape[:2]
    for idx, (x, y, w, h) in enumerate(word_boxes):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        word_img = image_original[y1:y2, x1:x2]
        word_images.append(word_img)

        if save_dir and debug:
            save_path = os.path.join(save_dir, f"word_{idx}.png")
            cv2.imwrite(save_path, word_img)

    return word_images




# === Нахождение строки ===
def find_text_lines(image_original, image_for_detection, padding=20, save_dir="cropped_lines", debug=True):

    if len(image_for_detection.shape) == 3:
        gray = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_for_detection.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(binary, axis=1)
    threshold = np.max(projection) * 0.1

    in_line = False
    start_y = 0
    line_coords = []

    for y, val in enumerate(projection):
        if val > threshold and not in_line:
            start_y = y
            in_line = True
        elif val <= threshold and in_line:
            if y - start_y >= 15:  
                line_coords.append((start_y, y))
            in_line = False
    if in_line:
        line_coords.append((start_y, len(projection)))

    if debug:
        os.makedirs(save_dir, exist_ok=True)

    height = image_original.shape[0]
    lines = []
    for idx, (start, end) in enumerate(line_coords):
        y1 = max(0, start - padding)
        y2 = min(height, end + padding)
        line_img = image_original[y1:y2, :]
        lines.append(line_img)
        if debug:
            cv2.imwrite(os.path.join(save_dir, f"line_{idx}.png"), line_img)

    return lines


# === Распознование слов в одной строке ===
def recognize_sentence(image_path):
    print(f"📷 Распознаю предложение на изображении: {image_path}")
    
    original_image_sentence = cv2.imread(image_path)
    cleaned_image_sentence = preprocess_for_line_detection(original_image_sentence)
    word_images = segment_sentence(image_original=original_image_sentence, image_for_detection=cleaned_image_sentence)
    if not word_images:
        return ""
    
    model = load_trained_model()
    dictionary = load_dictionary() 
    freq_dict = load_frequency_dictionary(FREQ_DICT_PATH)

    
    recognized_words = []
    for word_img in word_images:
        pil_image = Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
        
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            
            tensor = process_image(buffer)

        
        with torch.no_grad():
            output = model.predict(tensor)
            raw_text = indices_to_text(output[0])

        corrected_parts = []
        for part in raw_text.split():
            corrected = correct_word(part, dictionary, freq_dict)
            corrected_parts.append(corrected)
            print(f"  RAW → '{part}'   CORR → '{corrected}'")

        recognized_words.extend(corrected_parts)

    
    sentence = " ".join(recognized_words)

    return sentence


# === Распознование всех строк === 
def recognize_full_text(image_path):
    print(f"📄 Распознаю текст на изображении: {image_path}")

    original_image = cv2.imread(image_path)
    cleaned_image = preprocess_for_line_detection(original_image)

    lines = find_text_lines(image_original=original_image, image_for_detection=cleaned_image)
    if not lines:
        return ""

    model = load_trained_model()
    dictionary = load_dictionary()

    full_text = []

    for i, line_img in enumerate(lines):
        line_path = f"temp_line_{i}.png"
        cv2.imwrite(line_path, line_img)

        line_text = recognize_sentence(line_path)
        full_text.append(line_text)

    result = "\n".join(full_text)
    print(f"\n✅ Итоговый многострочный текст:\n{result}")
    return result





# === Запуск для теста ===
if __name__ == "__main__":
    test_image_path = "test_samples/test-text1.png"  
    result = recognize_full_text(test_image_path)
    print("📜 Итоговая строка:", result)