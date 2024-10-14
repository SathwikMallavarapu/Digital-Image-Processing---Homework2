import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image
import matplotlib.pyplot as plt

def kuwahara_filter(image, window_size=5):
    
    radius = window_size // 2
    padded_image = np.pad(image, pad_width=radius, mode='reflect')
    output_image = np.zeros_like(image)

    #applying the filter
    for i in range(radius, padded_image.shape[0] - radius):
        for j in range(radius, padded_image.shape[1] - radius):
            region_1 = padded_image[i - radius:i + 1, j - radius:j + 1] 
            region_2 = padded_image[i - radius:i + 1, j:j + radius + 1]  
            region_3 = padded_image[i:i + radius + 1, j - radius:j + 1]  
            region_4 = padded_image[i:i + radius + 1, j:j + radius + 1]  

           
            means = [np.mean(region) for region in [region_1, region_2, region_3, region_4]]
            variances = [np.var(region) for region in [region_1, region_2, region_3, region_4]]

            min_variance_index = np.argmin(variances)
            output_image[i - radius, j - radius] = means[min_variance_index]

    return output_image

# Load the image
img_1 = Image.open('C:\\Images\\image1.jpeg').convert('L')
img_array = np.array(img_1, dtype=float)

# Apply the Kuwahara filter
kuwahara_filtered_image = kuwahara_filter(img_array, window_size=5)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(kuwahara_filtered_image, cmap='gray')
plt.title('Kuwahara Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()
