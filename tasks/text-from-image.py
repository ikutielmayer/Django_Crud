import cv2
<<<<<<< HEAD
import pytesseract
from pytesseract import Output
import numpy as np

# Función para mostrar imágenes
def display(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Binarizacion
=======
import numpy as np
import pytesseract
from PIL import Image
import argparse
from matplotlib import pyplot as plt

# Función para redimensionar la imagen
def resize_image(image_path, max_width, max_height):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Calcular el factor de escala para ajustar la imagen al tamaño máximo permitido
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    # Calcular las nuevas dimensiones de la imagen
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Redimensionar la imagen usando interpolación bicúbica para mejorar la calidad
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Guardar la imagen redimensionada temporalmente
    resized_image_path = "tasks/templates/media/temp/resized_image.jpg"
    cv2.imwrite(resized_image_path, resized_img)

    return resized_image_path

# Funciones de procesamiento de imagen
>>>>>>> 92ffce644b08aa80654446c859ba0be827e20111
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
    image = cv2.medianBlur(image, 3)
    return image

def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return crop

<<<<<<< HEAD
# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar la imagen
image_file = "tasks/templates/media/pezula1.jpg"
img = cv2.imread(image_file)

# Procesamiento de la imagen
gray_image = grayscale(img)
thresh, im_bw = cv2.threshold(gray_image, 115, 255, cv2.THRESH_BINARY)
no_noise = noise_removal(im_bw)
eroded_image = thin_font(no_noise)
dilated_image = thick_font(eroded_image)
no_borders = remove_borders(dilated_image)

# Guardar la imagen procesada temporalmente
cv2.imwrite("tasks/templates/media/temp/no_borders.jpg", no_borders)

# Leer la imagen procesada para OCR
img = cv2.imread("tasks/templates/media/temp/no_borders.jpg")

# Extraer datos de la imagen utilizando OCR
data = pytesseract.image_to_data(img, lang='heb', output_type=Output.DICT)

# Contar palabras y letras
letter_count = 0

n_boxes = len(data['text'])
for i in range(n_boxes):
    if int(data['conf'][i]) > 0:  # Filtrar letras con confianza mayor a 0
        text = data['text'][i]
        if len(text) == 1 and text.isalnum():  # Verificar si es un solo carácter alfanumérico
            letter_count += 1

# Imprimir el conteo de letras
print(f"Total de letras: {letter_count}")

# Dibujar rectángulos alrededor de cada letra
for i in range(n_boxes):
    if int(data['conf'][i]) > 0:
        text = data['text'][i]
        if len(text) == 1 and text.isalnum():  # Verificar si es un solo carácter alfanumérico
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con cuadros alrededor de las letras
display(img, "Text with Letter Boxes")
=======
def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def display_resized(image_path):
    dpi = 80
    im_data = plt.imread(image_path)
    height, width = im_data.shape[:2]

    # Calcular el tamaño de la pantalla
    screen_width, screen_height = plt.figaspect(1)
    figsize = screen_width / dpi, screen_height / dpi

    # Crear la figura con el tamaño adecuado
    fig = plt.figure(figsize=figsize)

    # Añadir un eje para la imagen que ocupe toda la figura
    ax = fig.add_axes([0, 0, 1, 1])

    # Ocultar ejes y ticks
    ax.axis('off')

    # Mostrar la imagen redimensionada
    ax.imshow(im_data)

    # Ajustar la ventana de visualización para que quepa en la pantalla
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')  # Maximizar la ventana en Windows

    # Mostrar la figura
    plt.show()

# Configurar la ubicación de Tesseract OCR si es necesario
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Procesamiento de imagen y extracción de texto con Tesseract OCR')
    parser.add_argument('image_file', type=str, help='Ruta del archivo de imagen a procesar')

    # Parsear los argumentos
    args = parser.parse_args()

    # Procesamiento de la imagen original
    image_file = args.image_file
    img = cv2.imread(image_file)

    # Invertir imagen
    inverted_image = cv2.bitwise_not(img)
    cv2.imwrite("tasks/templates/media/temp/inverted.jpg", inverted_image)
    image_file_inverted = "tasks/templates/media/temp/inverted.jpg"

    # Convertir a escala de grises
    gray_image = grayscale(img)
    cv2.imwrite("tasks/templates/media/temp/gray.jpg", gray_image)
    image_file_gray = "tasks/templates/media/temp/gray.jpg"

    # Umbralización y eliminación de ruido
    thresh, im_bw = cv2.threshold(gray_image, 135, 255, cv2.THRESH_BINARY)
    cv2.imwrite("tasks/templates/media/temp/bw_image.jpg", im_bw)
    image_file_bw = "tasks/templates/media/temp/bw_image.jpg"

    no_noise = noise_removal(im_bw)
    cv2.imwrite("tasks/templates/media/temp/no_noise.jpg", no_noise)

    # Erosión y dilatación
    eroded_image = thin_font(no_noise)
    cv2.imwrite("tasks/templates/media/temp/eroded_image.jpg", eroded_image)

    dilated_image = thick_font(eroded_image)
    cv2.imwrite("tasks/templates/media/temp/dilated_image.jpg", dilated_image)

    # Eliminar bordes
    no_borders = remove_borders(dilated_image)
    cv2.imwrite("tasks/templates/media/temp/no_borders.jpg", no_borders)

    # Redimensionar la imagen procesada y mostrarla
    image_file = "tasks/templates/media/temp/no_borders.jpg"
    resized_image_path = resize_image(image_file, 1920, 1080)
    display_resized(resized_image_path)

    # Extraer texto con Tesseract OCR y guardar en archivo de texto
    extracted_text = pytesseract.image_to_string(Image.open('tasks/templates/media/temp/no_borders.jpg'), lang='heb', config='--psm 6')
    output_file = "tasks/templates/media/temp/extracted_text.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    print(f"Texto extraído guardado en: {output_file}")
>>>>>>> 92ffce644b08aa80654446c859ba0be827e20111
