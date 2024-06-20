import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont



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

# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the Guttman Stam font
# font_path = "tasks/templates/media/fonts/STAM.ttf"
# text_file = "tasks/templates/media/shema.txt"

# with open(text_file, "r", encoding="utf-8") as file:
#     text = file.read()

# font_size = 22
# image_size = (1000, 1500)

# # Generar la imagen con texto
# image = create_image_with_text(text, font_path, font_size, image_size)
# image.save("tasks/templates/media/temp/hebrew_text.png")

# Leer la imagen 
img = cv2.imread("tasks/templates/media/temp/im_bw.jpg")

# Preprocesamiento de imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Mostrar imagen preprocesada para verificación
# cv2.imshow("Preprocessed Image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Verificar si el paquete de idioma hebreo está disponible
# available_langs = pytesseract.get_languages(config='')
# print("Idiomas disponibles:", available_langs)

# # Asegurarse de que el hebreo esté en la lista de idiomas disponibles
# if 'heb' not in available_langs:
#     raise Exception("El paquete de idioma hebreo no está instalado en Tesseract.")

# Obtener datos de las palabras en hebreo
data = pytesseract.image_to_data(thresh, lang='heb', output_type=Output.DICT)

# Contar palabras y letras
word_count = 0
letter_count = 0

n_boxes = len(data['text'])
for i in range(n_boxes):
    if int(data['conf'][i]) > 1:  # Filtrar palabras con confianza mayor a 0
        word = data['text'][i]
        word_count += 1
        letter_count += len(word)

# Imprimir el conteo de palabras y letras
print(f"Total de palabras: {word_count}")
print(f"Total de letras: {letter_count}")

# Dibujar rectángulos alrededor de cada palabra
for i in range(n_boxes):
    if int(data['conf'][i]) > 15:
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 155, 0), 2)

# Mostrar imagen con cuadros alrededor de las palabras
cv2.imshow("Text with Word Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
