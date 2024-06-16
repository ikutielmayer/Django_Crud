import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont

# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Crear una imagen con texto en hebreo usando Guttman Stam
def create_image_with_text(text, font_path, font_size, image_size, line_spacing=10):
    image = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    
    lines = text.split('\n')
    y_text = 0

    for line in lines:
        # Invertir cada línea para que se renderice de derecha a izquierda
        line = line[::-1]
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = (image_size[0] - text_width - 10, y_text)
        draw.text(position, line, (0, 0, 0), font=font)
        y_text += text_height + line_spacing
        
    return image

# Path to the Guttman Stam font
font_path = "tasks/templates/media/fonts/STAM.ttf"
text_file = "tasks/templates/media/shema.txt"

with open(text_file, "r", encoding="utf-8") as file:
    text = file.read()

font_size = 22
image_size = (1000, 1500)

# Generar la imagen con texto
image = create_image_with_text(text, font_path, font_size, image_size)
image.save("tasks/templates/media/temp/hebrew_text.png")

# Leer la imagen generada
img = cv2.imread("tasks/templates/media/temp/hebrew_text.png")

# Preprocesamiento de imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Obtener cuadros de texto en hebreo
boxes = pytesseract.image_to_boxes(thresh, lang='heb')

# Dibujar rectángulos
for b in boxes.splitlines():
    b = b.split(' ')
    # Dibujar el cuadro delimitador de cada letra
    img = cv2.rectangle(
        img, (int(b[1]), img.shape[0] - int(b[2])),
        (int(b[3]), img.shape[0] - int(b[4])), (0, 255, 0), 1)

# Mostrar imagen con cuadros
cv2.imshow("Text with Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
