import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('C:\\Images\\image1.jpeg', 0)
rows, cols = image.shape

# Apply Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the Fourier Transform
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Fourier Transformed Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Function to create Butterworth filter
def butterworth_filter(image_shape, cutoff, order):
    rows, cols = image_shape
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - rows // 2, v - cols // 2, indexing='ij')  
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    return H


# Create Butterworth filter matching the image shape
butter_filter = butterworth_filter((rows, cols), cutoff=30, order=2)

# Apply the filter in the frequency domain
fshift_filtered = fshift * butter_filter

# Inverse Fourier Transform to bring back to the spatial domain
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the original and filtered images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Butterworth Filtered Image')
plt.show()


# Function to create Gaussian filter
def gaussian_filter(image_shape, sigma):
    rows, cols = image_shape
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - rows // 2, v - cols // 2, indexing='ij') 
    D = np.sqrt(U**2 + V**2)
    H = np.exp(-(D**2) / (2 * (sigma**2)))
    return H


# Create Gaussian filter matching the image shape
gauss_filter = gaussian_filter((rows, cols), sigma=10)

# Apply the Gaussian filter in the frequency domain
fshift_filtered_gauss = fshift * gauss_filter

# Display the original and filtered images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.show()
