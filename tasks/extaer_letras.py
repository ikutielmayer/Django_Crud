import cv2
import numpy as np
import os

def split_hebrew_letters(image_path, output_directory):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización para obtener solo las letras
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos de las letras
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterar sobre los contornos y guardar cada letra como imagen individual
    for i, contour in enumerate(contours):
        # Obtener el cuadro delimitador de la letra
        x, y, w, h = cv2.boundingRect(contour)
        
        # Recortar la región de la letra de la imagen original
        letter = image[y:y+h, x:x+w]
        
        # Guardar la letra como imagen individual en formato PNG
        output_path = os.path.join(output_directory, f"letter_{i}.png")
        cv2.imwrite(output_path, letter)
        
        print(f"Letra guardada en: {output_path}")

# Ruta de la imagen que contiene todas las letras juntas
input_image_path = "tasks/templates/media/temp/letras/no_borders.jpg"

# Directorio de salida para las letras individuales
output_directory = "tasks/templates/media/temp/letras/hebrew_letters"

# Separar las letras y guardarlas como imágenes individuales
split_hebrew_letters(input_image_path, output_directory)
