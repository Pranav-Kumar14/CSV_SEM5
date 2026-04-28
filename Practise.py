import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

# #KMeans
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png')
# image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# pixels = image_rgb.reshape((-1, 3))
# k = int(input("Enter the value of K : "))
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(pixels)
# segments = kmeans.cluster_centers_[kmeans.labels_]
# segmented_image = segments.reshape(image_rgb.shape).astype(np.uint8)
# segmented_image = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
# cv.imshow('Original Image', image)
# cv.imshow('Segmented Image', segmented_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# #Histogram Equalisation
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
# cdf = hist.cumsum()
# cdf_norm = cdf*255 / cdf[-1]
# equalised = np.interp(image.flatten(), bins[:-1], cdf_norm).reshape(image.shape).astype(np.uint8)
# cv.imshow('Original Image', image)
# cv.imshow('Equalised Image', equalised)
# cv.waitKey(0)
# cv.destroyAllWindows()

# #Histogram Matching
# def histo_matching(src, ref):
#     src_flat = src.flatten()
#     ref_flat = ref.flatten()
#     src_hist, _ = np.histogram(src_flat, bins=256, range = [0, 256])
#     ref_hist, _ = np.histogram(ref_flat, bins=256, range = [0, 256])
#     src_cdf = src_hist.cumsum()
#     ref_cdf = ref_hist.cumsum()
#     lookup = np.zeros(256, dtype=np.uint8)
#     ref_idx = 0
#     for src_idx in range(256):
#         while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
#             ref_idx += 1
#         lookup[src_idx] = ref_idx
#     matched = lookup[src_flat].reshape(src.shape).astype(np.uint8)
#     return matched
# image1 = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# image2 = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
# matched_image = histo_matching(image1, image2)
# cv.imshow('Source Image', image1)
# cv.imshow('Reference Image', image2)
# cv.imshow('Matched Image', matched_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # Otsu Thresholding
# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# def otsu(image):
#     hist, _ = np.histogram(image.flatten(), 256, [0, 256])
#     hist = hist.astype(np.float32)
#     hist_norm = hist / hist.sum()
#     cum_sum = np.cumsum(hist_norm)
#     cum_mean = np.cumsum(hist_norm * np.arange(256))
#     global_mean = cum_mean[-1]
#     bcw = (((global_mean * cum_sum) - cum_mean)**2)/ ((cum_sum * (1-cum_sum)) + 1e-7)
#     threshold = np.argmax(bcw)
#     max_bcw = bcw[threshold]
#     total_variance = np.sum(hist_norm * (np.arange(256) - global_mean) ** 2)
#     separability = max_bcw / (total_variance + 1e-7)
#     thresh_image = np.zeros_like(image)
#     thresh_image[image > threshold] = 255
#     return threshold, separability, thresh_image
# threshold, separability, binary_image = otsu(image)
# print(f"Otsu's Optimal Threshold: {threshold}")
# print(f"Otsu's Optimal Separability: {separability}")
# cv.imshow('Original Image', image)
# cv.imshow('Otsu Thresholding', binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

#Hough Transformation for Line Detection

def gaussian_kernel(size, sigma = 1):
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/ (2*sigma**2))
    return kernel/np.sum(kernel)
def gaussian_filter(image, size=5, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    pad = size//2
    padded_image = np.pad(image, pad)
    output_image = np.zeros_like(image, dtype = np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+size, j:j+size]
            output_image[i, j] = np.sum(window*kernel)
    return output_image
def sobel_gradient(image):
    Kx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)
    pad = 1
    padded_image = np.pad(image, pad)
    Gx = np.zeros_like(image, dtype=np.float32)
    Gy = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+3, j:j+3]
            Gx[i, j] = np.sum(window * Kx)
            Gy[i, j] = np.sum(window * Ky)
    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx) * (180. / np.pi)
    direction[direction < 0 ] += 180
    return magnitude, direction
def non_max(magnitude, direction):
    H, W = magnitude.shape
    Z = np.zeros((H, W), dtype = np.float32)
    angle = direction.copy()
    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            if (0<= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <=180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if (magnitude[i, j]>=q) and (magnitude[i, j]>=r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z
def threshold(image, lowRat=0.05, highRat=0.15):
    high = image.max() * highRat
    low = high * lowRat
    res = np.zeros_like(image, dtype=np.uint8)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image>=high)
    weak_i, weak_j = np.where((image<=high) & (image>=low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res
def hysteresis(image, weak=75, strong=255):
    H, W = image.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if image[i, j] == weak:
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                    or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                    or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image
def canny_edge(image):
    blurred = gaussian_filter(image, size=5, sigma=1)
    gradient_magnitude, gradient_direction = sobel_gradient(blurred)
    nonMaxImg = non_max(gradient_magnitude, gradient_direction)
    thresholdImg = threshold(nonMaxImg)
    edgeImg = hysteresis(thresholdImg)
    return edgeImg
def hough(edges, threshold=100):
    H, W = edges.shape
    diag_len = int(np.sqrt(H**2 + W**2))
    rhos = np.arange(-diag_len, diag_len + 1, 1)
    thetas = np.deg2rad(np.arange(0, 180, 1))
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(edges)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(round(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx])) + diag_len)
            acc[rho, t_idx] += 1
    lines = []
    for rho_idx in range(acc.shape[0]):
        for theta_idx in range(acc.shape[1]):
            if acc[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                lines.append((rho, theta))
    return lines, acc, rhos, thetas
def drawlines(image, lines):
    img_lines = image.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_lines
image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/sudoku.jpg', cv.IMREAD_GRAYSCALE)
edges = canny_edge(image)
lines, accumulator, rhos, thetas = hough(edges, threshold=150)
image_with_lines = drawlines(cv.cvtColor(image, cv.COLOR_GRAY2BGR), lines)
cv.imshow('Original Image', image)
cv.imshow('Canny Edges', edges)
cv.imshow('Hough Lines', image_with_lines)
cv.waitKey(0)
cv.destroyAllWindows()



