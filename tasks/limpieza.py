# import cv2
# import numpy as np
# import os

# # Función para mostrar imágenes
# def display(image, title="Image"):
#     cv2.imshow(title, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
# # Binarizacion
# def grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# def thin_font(image):
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2, 2), np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return image

# def thick_font(image):
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2, 2), np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return image

# def noise_removal(image):
#     kernel = np.ones((1, 1), np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     kernel = np.ones((1, 1), np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     image = cv2.medianBlur(image, 3)
#     return image

# def remove_borders(image):
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
#     cnt = cntsSorted[-1]
#     x, y, w, h = cv2.boundingRect(cnt)
#     crop = image[y:y+h, x:x+w]
#     return crop

# input_image_path = "tasks/templates/media/letras.jpg"
# img = cv2.imread(input_image_path)

# # Preprocesamiento de imagen
# gray_image = grayscale(img)
# thresh, im_bw = cv2.threshold(gray_image, 135, 255, cv2.THRESH_BINARY)
# no_noise = noise_removal(im_bw)
# eroded_image = thin_font(no_noise)
# dilated_image = thick_font(eroded_image)
# no_borders = remove_borders(dilated_image)

# # Guardar la imagen procesada temporalmente
# cv2.imwrite("tasks/templates/media/temp/letras/no_borders.jpg", no_borders)

# # Leer la imagen procesada para OCR
# img = cv2.imread("tasks/templates/media/temp/letras/no_borders.jpg")

# # Mostrar imagen preprocesada para verificación
# display(img, "Text with Letter Boxes")
import cv2
import numpy as np
import os

# Función para mostrar imágenes
def display(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Binarizacion
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return crop

def smooth_edges(image):
    # Aplicar un filtro Gaussiano para suavizar los bordes
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def split_hebrew_letters(image_path, output_directory):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = grayscale(image)
    
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
        
        # Suavizar los bordes de la letra
        letter_smoothed = smooth_edges(letter)
        
        # Guardar la letra como imagen individual en formato PNG
        output_path = os.path.join(output_directory, f"letter_{i}.png")
        cv2.imwrite(output_path, letter_smoothed)
        
        print(f"Letra guardada en: {output_path}")

# Ruta de la imagen que contiene todas las letras juntas
input_image_path = "tasks/templates/media/temp/letras/no_borders.jpg"

# Directorio de salida para las letras individuales
output_directory = "tasks/templates/media/temp/letras/hebrew_letters"

# Separar las letras y guardarlas como imágenes individuales
split_hebrew_letters(input_image_path, output_directory)

