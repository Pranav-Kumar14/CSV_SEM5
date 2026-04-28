import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern

# LBP function

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# radius = 7
# n_points = 8 * radius
# lbp_img = local_binary_pattern(image, n_points, radius, method='uniform')
# cv.imshow('Original Image', image)
# cv.imshow('LBP Image', lbp_img.astype(np.uint8))
# cv.waitKey(0)
# cv.destroyAllWindows()

# Harris Corner Detection

# import cv2 as cv
# import numpy as np
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
# image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# block_size = 2     
# ksize = 3         
# k = 0.04        
# dst = cv.cornerHarris(image, block_size, ksize, k)
# dst = cv.dilate(dst, None)
# threshold = 0.01 * dst.max()
# image_color[dst > threshold] = [0, 0, 255]
# cv.imshow('Harris Corners', image_color)
# cv.waitKey(0)
# cv.destroyAllWindows()

# FAST Corner Detection

# import cv2 as cv
# import numpy as np  
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
# image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# fast = cv.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
# keypoints = fast.detect(image, None)
# image_with_keypoints = cv.drawKeypoints(image_color, keypoints, None, color=(0, 255, 0))
# cv.imshow('FAST Keypoints', image_with_keypoints)
# cv.waitKey(0)
# cv.destroyAllWindows()

# SIFT Feature Detection and Description

# import cv2 as cv
# img1 = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# sift = cv.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)
# result_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv.imshow('SIFT Feature Matches', result_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# HOG Feature Descriptor

import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
features, hog_image = hog(image, 
                          orientations=9, 
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), 
                          block_norm='L2-Hys', 
                          visualize=True, 
                          transform_sqrt=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('HOG Visualization')
plt.axis('off')
plt.imshow(hog_image_rescaled, cmap='gray')
plt.show()


