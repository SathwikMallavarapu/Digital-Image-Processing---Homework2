import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image and convert to grayscale
img_1 = Image.open('C:\\Images\\image1.jpeg').convert('L')  
img_1_array = np.array(img_1, dtype=float) / 255.0  

# Floyd-Steinberg Dithering Algorithm
def floyd_steinberg_dithering(image):
    img_1 = image.copy()
    rows, cols = img_1.shape
    for i in range(rows):
        for j in range(cols):
            old_pixel = img_1[i, j]
            new_pixel = round(old_pixel)
            img_1[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            if j + 1 < cols:
                img_1[i, j + 1] += quant_error * 7 / 16
            if i + 1 < rows:
                if j > 0:
                    img_1[i + 1, j - 1] += quant_error * 3 / 16
                img_1[i + 1, j] += quant_error * 5 / 16
                if j + 1 < cols:
                    img_1[i + 1, j + 1] += quant_error * 1 / 16
    return np.clip(img_1 * 255, 0, 255).astype(np.uint8)  

# Jarvis-Judice-Ninke Dithering Algorithm
def jarvis_judice_ninke_dithering(image):
    img_1 = image.copy()
    rows, cols = img_1.shape

    kernel = np.array([[0, 0, 0, 7, 5],
                       [3, 5, 7, 5, 3],
                       [1, 3, 5, 3, 1]]) / 48

    for i in range(rows):
        for j in range(cols):
            old_pixel = img_1[i, j]
            new_pixel = round(old_pixel)
            img_1[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            for x in range(-2, 1):  
                for y in range(-2, 3):
                    if 0 <= i + x < rows and 0 <= j + y < cols:  
                        img_1[i + x, j + y] += quant_error * kernel[x + 2, y + 2]

    return np.clip(img_1 * 255, 0, 255).astype(np.uint8)

# Apply Floyd-Steinberg Dithering
dithered_img_fs = floyd_steinberg_dithering(img_1_array)

# Apply Jarvis-Judice-Ninke Dithering
dithered_img_jjn = jarvis_judice_ninke_dithering(img_1_array)

# Display both images for comparison
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(img_1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Floyd-Steinberg Dithering Algorithm Result
plt.subplot(1, 3, 2)
plt.imshow(dithered_img_fs, cmap='gray')
plt.title('Floyd-Steinberg Dithering Algorithm')
plt.axis('off')

# Jarvis-Judice-Ninke Dithering Algorithm Result
plt.subplot(1, 3, 3)
plt.imshow(dithered_img_jjn, cmap='gray')
plt.title('Jarvis-Judice-Ninke Dithering Algorithm')
plt.axis('off')

# Show the comparison
plt.tight_layout()
plt.show()
