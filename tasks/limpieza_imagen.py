import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt

# Function to display image using matplotlib
def cv2_imshow(img, title='Image'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

# Function to convert image to grayscale
def convert_gray(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale

# Function to threshold image and draw contours
def thresh_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_copy = img.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    return image_copy, contours

# Read an image in grayscale
img = cv2.imread('tasks/templates/media/mezuza_original.jpg', cv2.IMREAD_GRAYSCALE)

# Convert grayscale image to BGR (for display purposes)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Threshold and draw contours on the image
image_with_contours, contours = thresh_img(img_bgr)

# Save the image with contours
cv2.imwrite('tasks/templates/media/mezuza_contours.jpg', image_with_contours)

# Display original image and image with contours
if image_with_contours is not None:
    # Display images using matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not process the image.")

# Use pytesseract to extract text from the thresholded image with contours
# Specify the path to tesseract executable if needed
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'
custom_config = r'--oem 3 --psm 6 -c hocr_char_boxes=1'
extracted_text = pytesseract.image_to_string(image_with_contours, lang='heb', config=custom_config)

# Print the extracted text
print("Extracted Text:\n", extracted_text)

# Render the extracted text using a specific font
def render_text(text, font_path, font_size):
    # Create a blank image with white background
    img_pil = Image.new('RGB', (800, 200), color = (255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    
    # Load the font
    font = ImageFont.truetype(font_path, font_size)
    
    # Draw the text onto the image
    draw.text((10, 60), text, font=font, fill=(0, 0, 0))
    
    # Convert to OpenCV image format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_cv

# Path to the specific font you want to use
font_path = 'tasks/templates/media/fonts/STAM.TTF'
font_size = 48

# Render the extracted text
rendered_image = render_text(extracted_text, font_path, font_size)

# Save the rendered image
cv2.imwrite('tasks/templates/media/rendered_text.jpg', rendered_image)

# Display the rendered image
cv2_imshow(rendered_image, 'Rendered Text with Specific Font')

# Compare the images visually or by calculating similarity
def calculate_image_similarity(img1, img2):
    # Ensure the images have the same size
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute the Structural Similarity Index (SSI)
    from skimage.metrics import structural_similarity as ssim
    similarity_index, diff = ssim(gray1, gray2, full=True)
    
    return similarity_index

# Calculate similarity between the extracted text image and the rendered text image
similarity = calculate_image_similarity(image_with_contours, rendered_image)
print(f"Similarity between extracted and rendered text images: {similarity:.4f}")

# Display both images side by side for visual comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB))
plt.title('Rendered Text with Specific Font')
plt.axis('off')

plt.tight_layout()
plt.show()
