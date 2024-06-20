import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Función para mostrar imágenes usando matplotlib
def cv2_imshow(img, title='Image'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

# Función para convertir una imagen a escala de grises
def convert_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresh2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Función para umbralizar la imagen y dibujar contornos
def thresh_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_copy = img.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    return image_copy, contours

def pp(img):
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    results = pytesseract.image_to_data(thresh)

    for b in map(str.split, results.splitlines()[1:]):
        if len(b) == 12:
            x, y, w, h = map(int, b[6: 10])
            cv2.putText(img, b[11], (x, y + h + 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, 0)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    
    
# Función para resaltar palabras en la imagen
def highlight_words(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mi_config = '--psm 3 --oem 3'
    boxes = pytesseract.image_to_data(img_rgb, lang='heb', config=mi_config)
    image_copy = img.copy()
    
    for i, box in enumerate(boxes.splitlines()):
        if i == 0:
            continue
        box = box.split()
        if len(box) == 12:
            x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image_copy

# Lee la imagen en escala de grises
img = cv2.imread('tasks/templates/media/mezuza_original.jpg', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# Umbraliza la imagen y dibuja contornos
image_with_contours, contours = thresh_img(img_bgr)
cv2.imwrite('tasks/templates/media/mezuza_contours.jpg', image_with_contours)

tresh_imagen = thresh2(img_bgr)
# Resalta las palabras en la imagen
image_with_words = highlight_words(tresh_imagen)
cv2.imwrite('tasks/templates/media/mezuza_words.jpg', image_with_words)

# Muestra las imágenes
if image_with_contours is not None and image_with_words is not None:
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image_with_words, cv2.COLOR_BGR2RGB))
    plt.title('Image with Highlighted Words')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not process the image.")
