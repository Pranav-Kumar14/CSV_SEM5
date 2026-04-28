import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

# Binary Thresholding

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# threshold = int(input("Enter the threshold value (0-255) : "))
# output_image = np.zeros_like(image)
# output_image[image > threshold] = 255
# cv.imshow('Original Image', image)
# cv.imshow(f"Binary Thresholding with Threshold = {threshold}", output_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Otsu Thresholding

# def otsu_threshold(image):
#     hist, bins = np.histogram(image.flatten(), 256, [0,256])
#     hist = hist.astype(np.float32)
#     hist_norm = hist / hist.sum()
#     cumulative_sum = np.cumsum(hist_norm)
#     cumulative_mean = np.cumsum(hist_norm * np.arange(256))
#     global_mean = cumulative_mean[-1]
#     between_class_variance = (global_mean * cumulative_sum - cumulative_mean)**2 / (cumulative_sum * (1 - cumulative_sum) + 1e-7)
#     optimal_threshold = np.argmax(between_class_variance)
#     thresholded_image = np.zeros_like(image)
#     thresholded_image[image > optimal_threshold] = 255
#     return thresholded_image, optimal_threshold
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/ok_frame.png', cv.IMREAD_GRAYSCALE)
# binarized, threshold_val = otsu_threshold(image)
# print(f"Optimal Otsu Threshold: {threshold_val}")
# cv.imshow('Original Image', image)
# cv.imshow('Otsu Thresholded Image', binarized)
# cv.waitKey(0)
# cv.destroyAllWindows()

# K means Clustering for Image Segmentation

image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
pixels = image_rgb.reshape((-1, 3))
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_pixels.reshape(image_rgb.shape).astype(np.uint8)
segmented_image_bgr = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)

cv.imshow('Original Image', image)
cv.imshow(f'Segmented Image with {k} colors', segmented_image_bgr)
cv.waitKey(0)
cv.destroyAllWindows()

