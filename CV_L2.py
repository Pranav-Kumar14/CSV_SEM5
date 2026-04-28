import cv2 as cv
import numpy as np

# Negative of an Image

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# neg_img = 255-image
# cv.imshow('Original Image', image)
# cv.imshow('Negative Image', neg_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Log transform of an Image

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# log_image = np.log1p(image.astype(np.float32))
# log_image = log_image / np.max(log_image) * 255 
# log_image = np.uint8(log_image)
# cv.imshow('Original Image', image)
# cv.imshow('Log Transformed Image', log_image)
# cv.imshow('Original Image', image)
# cv.imshow('Log Transformed Image', log_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Gamma Transformation of an image
 
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# gammas = [0.1, 0.5, 0.7, 1, 1.5, 3, 5.0]
# for gamma in gammas : 
#     gamma_image = np.power(image.astype(np.float32)/255.0, gamma)
#     gamma_image = np.uint8(gamma_image * 255)
#     cv.imshow(f'Gamma Transformed Image (gamma={gamma})', gamma_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# cv.destroyAllWindows()

#Piecewise Linear Transformation

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def pixelVal(pix, r1, s1, r2, s2):
#     if (0<=pix and pix<=r1):
#         return (s1/r1)*pix
#     elif (r1<pix and pix<=r2):
#         return ((s2-s1)/(r2-r1))*(pix-r1) + s1
#     else:
#         return ((255-s2)/(255-r2))*(pix-r2) + s2
# r1 = 70
# s1 = 0
# r2 = 140
# s2 = 255
# pixelVec = np.vectorize(pixelVal)
# contrast_stretched = pixelVec(image, r1, s1, r2, s2)
# cv.imshow('Original Image', image)
# cv.imshow('Contrast Stretched Image', contrast_stretched)
# cv.waitKey(0)
# cv.destroyAllWindows()

#Histogram Equalization

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# hist, bins = np.histogram(image.flatten(), 256, [0, 256])
# cdf = hist.cumsum()
# cdf_normalised = cdf*255 / cdf[-1]
# equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalised)
# equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)
# cv.imshow('Original Image', image)
# cv.imshow('Equalized Image', equalized_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Histogra Matching

# def histogram_matching(source, ref):
#     src_flat = source.flatten()
#     ref_flat = ref.flatten()
#     src_hist, bins = np.histogram(src_flat, 256, [0, 256], density=True)
#     ref_hist, _ = np.histogram(ref_flat, 256, [0, 256], density=True)
#     src_cdf = src_hist.cumsum()
#     ref_cdf = ref_hist.cumsum()
#     lookup_table = np.zeros(256, dtype=np.uint8)
#     ref_idx = 0
#     for src_idx in range(256):
#         while ref_idx<255 and ref_cdf[ref_idx]<src_cdf[src_idx]:
#             ref_idx+=1
#         lookup_table[src_idx] = ref_idx
#     matched = lookup_table[src_flat]
#     matched = matched.reshape(source.shape).astype(np.uint8)
#     return matched
# ref_image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# source_image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/ok_frame.png', cv.IMREAD_GRAYSCALE)
# matched_image = histogram_matching(source_image, ref_image)
# cv.imshow('Source Image', source_image)
# cv.imshow('Reference Image', ref_image)
# cv.imshow('Matched Image', matched_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Resizing and Cropping

image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png')
fx = float(input("Enter scaling factor along x-axis: "))
fy = float(input("Enter scaling factor along y-axis: "))
resized_image = cv.resize(image, (0, 0), fx=fx, fy=fy)
crop_size = int(input("Enter the crop size (square crop): "))
h, w = resized_image.shape[:2]
start_x = max(w//2 - crop_size//2, 0)
start_y = max(h//2 - crop_size//2, 0)
end_x = min(start_x + crop_size, w)
end_y = min(start_y + crop_size, h)
cropped_image = resized_image[start_y:end_y, start_x:end_x]
cv.imshow('Original Image', image)
cv.imshow('Resized Image', resized_image)
cv.imshow('Cropped Image', cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()


