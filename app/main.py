import os
import re
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)

model_path = 'models/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.25
model.iou = 0.45
model.max_det = 1000
model.eval()

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='pt', show_log=False)

SAVE_PATH = "processed_images"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def save_image(image, filename):
    if image is not None and image.size > 0:
        path = os.path.join(SAVE_PATH, filename)
        cv2.imwrite(path, image)

def detect_plate(img):
    if img is None or img.size == 0:
        logging.error("Imagem vazia ou inválida fornecida para detecção.")
        return None

    img_resized = cv2.resize(img, (640, 640))

    results = model(img_resized, size=640)

    results.show() 
    crops = results.crop(save=False)
    if len(crops) > 0:
        plate_region = crops[0]['im']
        return plate_region

    logging.warning("Nenhuma placa detectada na imagem.")
    return None



def post_process_plate_text(plate_text):
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

            if idx < 3:
                if char in '012568{':
                    corrected_char = correction_map.get(char, char).upper()

           
            elif idx == 3 or idx == 5:
                if char in 'OISBGR{': 
                    corrected_char = correction_map.get(char, char)

          
            elif idx == 4:
                if char in '012568{': 
                    corrected_char = correction_map.get(char, char).upper()

            else:
                if char in 'OISBGR{': 
                    corrected_char = correction_map.get(char, char)

            corrected_plate.append(corrected_char)
        return ''.join(corrected_plate)

    corrected_plate = apply_correction(plate_text)

    logging.info(f"Original plate text: {plate_text}")
    logging.info(f"Corrected plate text: {corrected_plate}")

    if mercosur_pattern.match(corrected_plate) or old_pattern.match(corrected_plate):
        return corrected_plate

    logging.warning("Corrected plate does not match any pattern, but returning corrected version.")
    return corrected_plate


def perform_paddle_ocr(image):
    try:
        results = paddle_ocr.ocr(image, det=False, cls=True)

        if not results or not isinstance(results, list):
            logging.error("PaddleOCR returned an unexpected result or empty result.")
            return ""

        plate_text, ocr_confidence = max(
            results,
            key=lambda ocr_prediction: max(
                ocr_prediction,
                key=lambda ocr_prediction_result: ocr_prediction_result[1],  # type: ignore
            ),
        )[0]

        plate_text_filtered = re.sub(r"[^A-Z0-9- ]", "", plate_text).strip("- ")

        return plate_text_filtered

    except Exception as e:
        logging.error(f"Error performing PaddleOCR: {str(e)}")
        return ""


def preprocess_image(src):
    normalize = cv2.normalize(
        src, np.zeros((src.shape[0], src.shape[1])), 0, 255, cv2.NORM_MINMAX
    )
    denoise = cv2.fastNlMeansDenoisingColored(
        normalize, h=10, hColor=10, templateWindowSize=7, searchWindowSize=15
    )
    grayscale = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    save_image(threshold, "processed_plate.png")  # Save processed image for debugging
    return threshold
    


def apply_ocr(plate_region):
    if plate_region is None or plate_region.size == 0:
        logging.info("Empty or invalid plate region provided for OCR.")
        return None

    try:
        processed_image = preprocess_image(plate_region)
        if processed_image is None or processed_image.size == 0:
            logging.info("Processed image is invalid.")
            return None

        plate_text = perform_paddle_ocr(processed_image)
        if plate_text:
            logging.info(f"Detected text: {plate_text}")
            return plate_text

        logging.info("No text detected.")
        return None
    except Exception as e:
        app.logger.error(f"Error processing plate: {str(e)}")
        return None
    

@app.route('/read-plate', methods=['POST'])
def read_plate():
    logging.info("Received request to /read-plate")
    try:
        if 'image' not in request.files:
            logging.info("No image provided in the request.")
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image_path = os.path.join("/tmp", image_file.filename)
        image_file.save(image_path)

        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            logging.info("Invalid image provided.")
            return jsonify({'error': 'Invalid image provided'}), 400

       # save_image(img, "original_image.png")

        plate_region = detect_plate(img)
        if plate_region is None:
            logging.info("No plate detected in the image.")
            return jsonify({'error': 'Plate not detected'}), 404

        plate_text = apply_ocr(plate_region)
        if plate_text:
            processed_plate = post_process_plate_text(plate_text)
            return jsonify({'plate': processed_plate})

        logging.debug("No text detected on the plate.")
        return jsonify({'error': 'Plate not detected'}), 404
    except Exception as e:
        logging.info(f"Internal server error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
