import cv2
import pytesseract



# Gray Scale
def convert_grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Noise removal
def blur(img, param):
    img = cv2.medianBlur(img, param)
    return img

# Thresholding
def threshold(img):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Cargar imagen
img = cv2.imread("tasks/templates/media/temp/bw_image.jpg")
h, w, c = img.shape

# Preprocesamiento de imagen
gray = convert_grayscale(img)
blurred = blur(gray, 3)
thresh = threshold(blurred)

# Obtener cuadros de texto
boxes = pytesseract.image_to_boxes(thresh, lang='heb')

# Dibujar rectángulos
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(
        img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# Mostrar imagen con cuadros
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
