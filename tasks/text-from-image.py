import cv2
import pytesseract
from pytesseract import Output
import numpy as np

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
