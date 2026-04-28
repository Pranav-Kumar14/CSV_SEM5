import cv2 as cv
import numpy as np

def gaussian_kernel(size, sigma=1):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(image, pad_size, mode='reflect')
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
    angle = direction
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

def threshold(image, lowRatio=0.05, highRatio=0.15):
    highThreshold = image.max() * highRatio
    lowThreshold = highThreshold * lowRatio
    res = np.zeros_like(image, dtype=np.uint8)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image >= lowThreshold) & (image < highThreshold))
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

def hough_transform(edges, threshold=100):
    H, W = edges.shape
    diag_len = int(np.sqrt(H**2 + W**2))
    rhos = np.arange(-diag_len, diag_len + 1, 1)
    thetas = np.deg2rad(np.arange(0, 180, 1))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)
    y_idxs, x_idxs = np.nonzero(edges)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + diag_len
            accumulator[rho, t_idx] += 1

    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                lines.append((rho, theta))
    return lines, accumulator, rhos, thetas

def draw_lines(img, lines):
    img_lines = img.copy()
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

edges = canny_edge_detection(image)
while True:
    try:
        threshold_input = int(input("Enter the Hough Transform threshold value (e.g., 100): "))
        if threshold_input <= 0:
            print("Please enter a positive integer.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")
lines, accumulator, rhos, thetas = hough_transform(edges, threshold_input)
final_image = draw_lines(cv.cvtColor(image, cv.COLOR_GRAY2BGR), lines)
cv.imshow('Original Image', image)
cv.imshow('Canny Edges', edges)
cv.imshow('Detected Lines', final_image)
cv.waitKey(0)
cv.destroyAllWindows()

