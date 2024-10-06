import os
import torch
import cv2
import easyocr
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Carregar o modelo YOLOv5 treinado para detecção
model_path = 'models/best.pt'  # Atualize o caminho para o modelo baixado localmente
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # Carregar o modelo YOLOv5 diretamente
model.eval()  # Colocar o modelo em modo de avaliação

app = Flask(__name__)

# Definir o diretório para salvar imagens processadas
SAVE_PATH = "processed_images"

# Verificar e criar o diretório se não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Carregar o OCR (EasyOCR e Tesseract)
reader = easyocr.Reader(['en'])

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
    results = model(img_resized)

    # Log dos resultados
    detections = results.pandas().xyxy[0]  # Obter resultados do modelo em um DataFrame
    logging.info(f"Resultados da detecção: {detections}")

    if len(detections) == 0:
        logging.warning("Nenhuma detecção encontrada na imagem.")

    img_with_detections = img.copy()  # Copiar imagem para desenhar todas as detecções

    for idx, row in detections.iterrows():
        if row['name'] == 'license_plate':  # Verificar se o objeto detectado é uma placa
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            # Garantir que as coordenadas estão dentro dos limites da imagem original
            x1 = max(0, min(x1, img.shape[1] - 1))
            y1 = max(0, min(y1, img.shape[0] - 1))
            x2 = max(0, min(x2, img.shape[1] - 1))
            y2 = max(0, min(y2, img.shape[0] - 1))

            # Ajustar coordenadas para garantir que a região não seja vazia
            if y2 <= y1:
                y2 = y1 + 1
            if x2 <= x1:
                x2 = x1 + 1

            logging.info(f"Coordenadas da placa detectada: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            plate_region = img[y1:y2, x1:x2]
            save_image(plate_region, f"plate_region_{idx}.png")  # Salvar cada região detectada para debug

            # Desenhar retângulo ao redor da placa detectada na imagem completa
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Salvar imagem com todas as detecções destacadas
    save_image(img_with_detections, "detections_full.png")

    return img_with_detections

# Função para aplicar OCR na imagem da placa
def apply_ocr(plate_region):
    if plate_region is None or plate_region.size == 0:
        logging.error("Imagem da placa vazia ou inválida fornecida para OCR.")
        return None

    # Usar tanto EasyOCR quanto Tesseract para tentar reconhecer a placa
    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    save_image(gray_plate, "gray_plate.png")
    results_easyocr = reader.readtext(gray_plate)
    if results_easyocr:
        plate_text = ' '.join([res[1] for res in results_easyocr])
        logging.info(f"Texto detectado pelo EasyOCR: {plate_text}")
        return plate_text

    # Caso o EasyOCR não funcione bem, tentar com Tesseract
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    plate_text = pytesseract.image_to_string(gray_plate, config=config).strip()
    logging.info(f"Texto detectado pelo Tesseract: {plate_text}")
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
            return jsonify({'plate': plate_text})

        logging.warning("Nenhum texto detectado na placa.")
        return jsonify({'error': 'Plate not detected'}), 404

    except Exception as e:
        logging.error(f"Erro interno do servidor: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)