import cv2
import numpy as np

def compare_images(image1_path, image2_path):
    # Leer las imágenes
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Inicializar el detector ORB
    orb = cv2.ORB_create()

    # Detectar características y descriptores
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Crear el objeto de coincidencia de características BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Realizar la coincidencia de características
    matches = bf.match(descriptors1, descriptors2)

    # Ordenar las coincidencias por distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Calcular el porcentaje de coincidencia
    good_matches = [m for m in matches if m.distance < 70]  # Umbral de distancia
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0

    # Dibujar las coincidencias
    result_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar la imagen de resultado
    cv2.imshow('Matches', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return match_percentage

# Rutas de las imágenes de letras
image1_path = "path/to/letter1.png"
image2_path = "path/to/letter2.png"

# Comparar las imágenes y obtener el porcentaje de coincidencia
similarity_percentage = compare_images("tasks/templates/media/temp/letras/hebrew_letters/ayin.png", "tasks/templates/media/temp/letras/hebrew_letters/letra2.png")
print(f"Las imágenes son similares en un {similarity_percentage:.2f}%")
