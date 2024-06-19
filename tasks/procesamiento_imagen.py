import cv2
import sys
import matplotlib.pyplot as plt

def main(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Verificar que la imagen se haya cargado correctamente
    if image is None:
        print("Error al cargar la imagen.")
        return

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mejorar la calidad de la imagen - aplicar aumento de contraste
    enhanced_image = cv2.equalizeHist(gray_image)

    # Eliminar ruido utilizando la función fastNlMeansDenoising
    denoised_image = cv2.fastNlMeansDenoisingMulti(enhanced_image, None, 7, 7, 30)

    # Mostrar la imagen original y la imagen preprocesada
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Escala de grises")
    plt.imshow(gray_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Mejorada y sin ruido")
    plt.imshow(denoised_image, cmap='gray')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <ruta_de_la_imagen>")
    else:
        main(sys.argv[1])
