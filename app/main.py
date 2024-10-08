import os
import torch
import cv2
import easyocr
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import logging
import re
from skimage import measure

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Carregar o modelo YOLOv5 treinado para detecção
model_path = 'models/best.pt'  # Atualize o caminho para o modelo baixado localmente
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # Carregar o modelo YOLOv5 diretamente

# Ajustar parâmetros do modelo
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

model.eval()  # Colocar o modelo em modo de avaliação

app = Flask(__name__)

# Definir o diretório para salvar imagens processadas
SAVE_PATH = "processed_images"

# Verificar e criar o diretório se não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Carregar o OCR (EasyOCR e Tesseract)
reader = easyocr.Reader(['pt'])

# Função para salvar a imagem
def save_image(image, filename):
    if image is not None and image.size > 0:  # Verificar se a imagem não está vazia
        path = os.path.join(SAVE_PATH, filename)
        cv2.imwrite(path, image)

# Função para detectar a placa usando YOLOv5
def detect_plate(img):
    if img is None or img.size == 0:
        logging.error("Imagem vazia ou inválida fornecida para detecção.")
        return None

    # Redimensionar a imagem para o tamanho esperado pelo modelo
    img_resized = cv2.resize(img, (640, 640))
    save_image(img_resized, "resized_image.png")  # Salvar imagem redimensionada para debug

    # Executar o modelo diretamente na imagem
    results = model(img_resized, size=640)

    # Salvar resultados
    results.show()  # Mostrar resultados

    # Recortar a placa detectada usando a função crop
    crops = results.crop(save=False)
    if len(crops) > 0:
        plate_region = crops[0]['im']  # Usar a primeira região recortada
        save_image(plate_region, "plate_region_cropped.png")  # Salvar a região da placa recortada
        return plate_region

    logging.warning("Nenhuma placa detectada na imagem.")
    return None


def post_process_plate_text(plate_text):
    # Mapeamento de confusões comuns entre números e letras
    correction_map = {
        '0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B', '{': 'R',
        'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8', 'R': 'R'
    }

    # Padrão para placas Mercosul (3 letras + 1 número + 1 letra + 2 números)
    mercosur_pattern = re.compile(r'^[A-Z]{3}[0-9]{1}[A-Z]{1}[0-9]{2}$')
    # Padrão para placas antigas (3 letras + 4 números)
    old_pattern = re.compile(r'^[A-Z]{3}[0-9]{4}$')

    def apply_correction(plate_text):
        corrected_plate = []
        for idx, char in enumerate(plate_text):
            corrected_char = char

            # Primeiros 3 caracteres devem ser letras
            if idx < 3:
                if char in '012568{':  # Corrigir números e '{' que parecem letras
                    corrected_char = correction_map.get(char, char).upper()

            # Posição 3 e 5 devem ser números (mercosul)
            elif idx == 3 or idx == 5:
                if char in 'OISBGR{':  # Corrigir letras e '{' que parecem números
                    corrected_char = correction_map.get(char, char)

            # Posição 4 deve ser letra, aplicar correção especial para "6" -> "G"
            elif idx == 4:
                if char == '6':
                    corrected_char = 'G'  # Substituir sempre o "6" por "G"
                elif char in '012568{':  # Aplicar as correções de letras/números
                    corrected_char = correction_map.get(char, char).upper()

            # Restante deve ser número
            else:
                if char in 'OISBGR{':  # Corrigir letras mal reconhecidas e '{'
                    corrected_char = correction_map.get(char, char)

            corrected_plate.append(corrected_char)
        return ''.join(corrected_plate)

    corrected_plate = apply_correction(plate_text)

    # Log para verificação
    logging.info(f"Original plate text: {plate_text}")
    logging.info(f"Corrected plate text: {corrected_plate}")

    # Se a placa corrigida não coincidir com o padrão Mercosul ou antigo, retornar a corrigida em vez da original
    if mercosur_pattern.match(corrected_plate) or old_pattern.match(corrected_plate):
        return corrected_plate

    # Caso não corresponda a nenhum padrão, ainda retornamos a placa corrigida
    logging.warning("Corrected plate does not match any pattern, but returning corrected version.")
    return corrected_plate

def preprocess_character_image(char_image):
    """
    Preprocess the character image for better OCR recognition.
    This includes resizing, thresholding, and noise reduction steps.
    """
    # Resize to a consistent size for better OCR (keeping the aspect ratio)
    resized_char = cv2.resize(char_image, (20, 40))

    # Convert to grayscale (if not already)
    if len(resized_char.shape) == 3:
        resized_char = cv2.cvtColor(resized_char, cv2.COLOR_BGR2GRAY)



    return resized_char

def segment_characters(plate_region):
    if plate_region is None or plate_region.size == 0:
        logging.error("Imagem da placa vazia ou inválida fornecida para segmentação de caracteres.")
        return []

    # Convert to grayscale and apply Otsu's thresholding
    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_image(thresh, "thresh_plate.png")  # Save thresholded image for debugging

    # Apply some morphological operations to improve segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    save_image(thresh, "thresh_morph.png")

    # Perform connected components analysis to segment potential characters
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype="uint8")

    # Loop over the unique components
    for label in np.unique(labels):
        # Ignore the background label
        if label == 0:
            continue

        # Create a label mask to show the connected component for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        # Find contours in the label mask
        cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            # Adjusted filtering conditions for narrower characters like "I"
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 1.0 and h > 15 and h < 0.95 * plate_region.shape[0]:
                charCandidates[y:y + h, x:x + w] = thresh[y:y + h, x:x + w]

    # Find final contours after filtering and sort them by x-coordinate (left to right)
    final_contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(final_contours, key=lambda c: cv2.boundingRect(c)[0])

    character_images = []

    for idx, c in enumerate(sorted_contours):
        (x, y, w, h) = cv2.boundingRect(c)

        # Ensure small characters like "I" are captured by adjusting the height threshold
        if h / plate_region.shape[0] > 0.35 and h < 0.95 * plate_region.shape[0]:
            char_image = thresh[y:y + h, x:x + w]
            char_image_preprocessed = preprocess_character_image(char_image)
            character_images.append(char_image_preprocessed)
            save_image(char_image_preprocessed, f"char_segment_preprocessed_{idx}.png")  # Save segmented characters

    if not character_images:
        logging.warning("Nenhum caractere segmentado da placa.")

    return character_images

def apply_ocr(plate_region, ocr_type='easyocr'):
    if plate_region is None or plate_region.size == 0:
        logging.error("Imagem da placa vazia ou inválida fornecida para OCR.")
        return None

    # Segmentar caracteres da placa
    character_images = segment_characters(plate_region)
    plate_text = ""

    # Iterate over each segmented character
    for idx, char_image in enumerate(character_images):
        if char_image is not None and char_image.size > 0:
            # Apply EasyOCR or Tesseract based on the 'ocr_type' argument
            if ocr_type == 'easyocr':
                result = reader.readtext(char_image, detail=0)
                char_text = ''.join(result).strip()
            elif ocr_type == 'tesseract':
                char_text = pytesseract.image_to_string(char_image, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
            else:
                logging.error(f"Tipo de OCR inválido: {ocr_type}")
                return None

            # Log and save the recognized character for debugging
            if char_text:
                logging.info(f"Character {idx}: {char_text}")
                plate_text += char_text
            else:
                logging.warning(f"No character detected for segment {idx}.")
        else:
            logging.warning(f"Empty or invalid character segment at index {idx}")

    if len(plate_text) < 7:
        logging.warning(f"Partial plate recognized: {plate_text} (expected 7 characters)")
    else:
        logging.info(f"Complete plate recognized: {plate_text}")

    return plate_text


@app.route('/read-plate', methods=['POST'])
def read_plate():
    try:
        if 'image' not in request.files:
            logging.error("Nenhuma imagem fornecida na requisição.")
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image_path = os.path.join("/tmp", image_file.filename)
        image_file.save(image_path)

        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            logging.error("Imagem inválida fornecida.")
            return jsonify({'error': 'Invalid image provided'}), 400

        save_image(img, "original_image.png")  # Salvar imagem original para debug

        # Detectar a placa
        plate_region = detect_plate(img)
        if plate_region is None:
            logging.warning("Nenhuma placa detectada na imagem.")
            return jsonify({'error': 'Plate not detected'}), 404

        # Aplicar OCR na placa detectada
        plate_text = apply_ocr(plate_region)

        if plate_text:
            # **Aqui garantimos que o post_process_plate_text seja chamado e aplicado ao resultado**
            logging.info(f"Raw plate text before processing: {plate_text}")
            
            processed_plate = post_process_plate_text(plate_text)
            
            logging.info(f"Processed plate text: {processed_plate}")
            
            return jsonify({'plate': processed_plate})

        logging.warning("Nenhum texto detectado na placa.")
        return jsonify({'error': 'Plate not detected'}), 404

    except Exception as e:
        logging.error(f"Erro interno do servidor: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
