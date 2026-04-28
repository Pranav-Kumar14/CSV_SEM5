import cv2 as cv
import numpy as np
import math

# Moving Average Filter

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def moving_average(image, kernel_size):
#     pad_size = kernel_size//2
#     padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
#     filtered_image = np.zeros_like(image)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             window = padded_image[i:i+kernel_size, j:j+kernel_size]
#             filtered_image[i, j] = np.mean(window)
#     return np.uint8(filtered_image)
# kernel_size = int(input("Enter the kernel size : "))
# smoothed_image = moving_average(image, kernel_size)
# cv.imshow('Original Image', image)
# cv.imshow('Smoothed Image', smoothed_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Median Filter

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def moving_average(image, kernel_size):
#     pad_size = kernel_size//2
#     padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
#     filtered_image = np.zeros_like(image)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             window = padded_image[i:i+kernel_size, j:j+kernel_size]
#             filtered_image[i, j] = np.median(window)
#     return np.uint8(filtered_image)
# kernel_size = int(input("Enter the kernel size : "))
# smoothed_image = moving_average(image, kernel_size)
# cv.imshow('Original Image', image)
# cv.imshow('Smoothed Image', smoothed_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Gaussian Filter

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def gaussian_kernel(size, sigma):
#     ax = np.arange(-size//2 + 1. , size//2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2)/(2. *sigma**2))
#     return kernel / np.sum(kernel)
# def gaussian_filter(image, kernel_size, sigma):
#     pad_size = kernel_size//2
#     padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
#     kernel = gaussian_kernel(kernel_size, sigma)
#     filtered_image = np.zeros_like(image)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             window = padded_image[i:i+kernel_size, j:j+kernel_size]
#             filtered_image[i, j] = np.sum(window * kernel)
#     return np.uint8(filtered_image)
# kernel_size = int(input("Enter the kernel size : "))
# sigma = float(input("Enter the sigma value : "))
# smoothed_image = gaussian_filter(image, kernel_size, sigma)
# cv.imshow('Original Image', image)
# cv.imshow('Smoothed Image', smoothed_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Unmark Sharping

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def gaussian_kernel(size, sigma):
#     ax = np.arange(-size//2 + 1. , size//2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2)/(2. *sigma**2))
#     return kernel / np.sum(kernel)
# def gaussian_filter(image, kernel_size, sigma):
#     pad_size = kernel_size//2
#     padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
#     kernel = gaussian_kernel(kernel_size, sigma)
#     filtered_image = np.zeros_like(image)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             window = padded_image[i:i+kernel_size, j:j+kernel_size]
#             filtered_image[i, j] = np.sum(window * kernel)
#     return np.uint8(filtered_image)
# def unsharp_masking(image, kernel_size, sigma, amount):
#     blurred = gaussian_filter(image, kernel_size, sigma)
#     mask = image.astype(np.float32) - blurred.astype(np.float32)
#     sharpened = image.astype(np.float32) + amount * mask
#     sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
#     return sharpened
# kernel_size = int(input("Enter the kernel size : "))
# sigma = float(input("Enter the sigma value : "))
# amount = float(input("Enter the amount value : "))
# sharpened_image = unsharp_masking(image, kernel_size, sigma, amount)
# cv.imshow('Original Image', image)
# cv.imshow('Sharpened Image', sharpened_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Apply Sobel Filter

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def sobel_filter(image):
#     Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#     Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
#     pad_size = 1
#     padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
#     Gx = np.zeros_like(image, dtype=np.float32)
#     Gy = np.zeros_like(image, dtype=np.float32)
#     h, w = image.shape
#     for i in range(h):
#         for j in range(w):
#             window = padded_image[i:i+3, j:j+3]
#             Gx[i, j] = np.sum(window * Kx)
#             Gy[i, j] = np.sum(window * Ky)
#     G = np.sqrt(Gx**2 + Gy**2)
#     G = (G/G.max()*255).astype(np.uint8)
#     return G
# edge_image = sobel_filter(image)
# cv.imshow('Original Image', image)
# cv.imshow('Edge Detected Image', edge_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Image Gradient 

# def compute_gradient(image):
#     Kx = np.array([[-1, 0, 1],
#                    [-1, 0, 1],
#                    [-1, 0, 1]], dtype=np.float32)
#     Ky = np.array([[ 1,  1,  1],
#                    [ 0,  0,  0],
#                    [-1, -1, -1]], dtype=np.float32)
#     pad_size = 1
#     padded_img = np.pad(image, pad_size, mode='constant', constant_values=0)
#     Gx = np.zeros_like(image, dtype=np.float32)
#     Gy = np.zeros_like(image, dtype=np.float32)
#     height, width = image.shape
#     for i in range(height):
#         for j in range(width):
#             window = padded_img[i:i+3, j:j+3]
#             Gx[i, j] = np.sum(Kx * window)
#             Gy[i, j] = np.sum(Ky * window)
#     gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
#     gradient_direction = np.arctan2(Gy, Gx) * (180 / np.pi)
#     gradient_direction = (gradient_direction + 360) % 360 
#     mag_norm = (gradient_magnitude / gradient_magnitude.max()) * 255
#     return mag_norm.astype(np.uint8), gradient_direction
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# magnitude, direction = compute_gradient(image)
# cv.imshow('Original Image', image)
# cv.imshow('Gradient Magnitude', magnitude)
# direction_vis = np.uint8((direction / 360) * 255)
# cv.imshow('Gradient Direction', direction_vis)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Canny Edge Detection

def gaussian_kernel(size, sigma=1):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
def convolve(image, kernel):
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(window * kernel)
    return output
def sobel_gradients(image):
    Kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]], dtype=np.float32)
    Gx = convolve(image, Kx)
    Gy = convolve(image, Ky)
    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx) * (180 / np.pi)
    direction[direction < 0] += 180
    return magnitude, direction
def non_maximum_suppression(magnitude, direction):
    H, W = magnitude.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = direction.copy()
    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0
    return Z
def threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    res = np.zeros_like(image, dtype=np.uint8)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong
def hysteresis(img, weak, strong=255):
    H, W = img.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img
def canny_edge_detection(image, kernel_size=5, sigma=1.4, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = convolve(image, kernel)
    magnitude, direction = sobel_gradients(blurred)
    nonMaxImg = non_maximum_suppression(magnitude, direction)
    threshImg, weak, strong = threshold(nonMaxImg, lowThresholdRatio, highThresholdRatio)
    edge_img = hysteresis(threshImg, weak, strong)
    return edge_img
image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/ok_frame.png', cv.IMREAD_GRAYSCALE)
edges = canny_edge_detection(image)
cv.imshow('Original Image', image)
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)
cv.destroyAllWindows()

