import cv2 as cv
import numpy as np

# Question 1

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png', cv.IMREAD_GRAYSCALE)
# cv.imshow('Image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.imwrite('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao_gray.png', image)

# Question 2

# cap = cv.VideoCapture('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/ok.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Reached End")
#         break
#     cv.imshow('Video', frame)
#     if cv.waitKey(27) & 0xFF == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()
# cv.imwrite('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/ok_frame.png', frame)

# Question 3

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png')
# h, w, _ = image.shape
# x = int(input(f"Enter the X coordinate of the pixel (between 0 and {w-1}) : "))
# y = int(input(f"Enter the Y coordinate of the pixel (between 0 and {h-1}) : "))
# if 0<=x<w and 0<=y<h:
#     b, g, r = image[y, x]
#     print(f"The pixel at coordinates : {x}, {y} has the RGB value : {r}, {g}, {b}")
# else:
#     print("Coordinates out of bounds")
# cv.destroyAllWindows()

# Question 4

# img_h, img_w = 500, 500
# image = np.ones((img_h, img_w, 3), dtype = np.uint8)*255
# x = int(input(f"Enter the rectangle width (0 to {img_w-1}) : "))
# y = int(input(f"Enter the rectangle height (0 to {img_h-1}) : "))
# r = int(input("Enter the rectangle red value (0 to 255) : "))
# g = int(input("Enter the rectangle green value (0 to 255) : "))
# b = int(input("Enter the rectangle blue value (0 to 255) : "))
# start_x = (img_w - x) // 2
# start_y = (img_h - y) // 2
# end_x = start_x + x
# end_y = start_y + y
# cv.rectangle(image, (start_x, start_y), (end_x, end_y), (b, g, r), thickness = -1)
# cv.imshow('Goddahm Rectangle', image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Question 5

# image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png')
# og_h, og_w, _ = image.shape
# new_h = int(input(f"Enter the new height : "))
# new_w = int(input(f"Enter the new width : "))
# cv.imshow('Original Image', image)
# resized_image = cv.resize(image, (new_w, new_h))
# cv.imshow('Resized Image', resized_image)
# cv.waitKey(0)  
# cv.destroyAllWindows()

# Question 6
image = cv.imread('/Users/pranavkumar/Desktop/Coding/Python/BDA/images/lmao.png')
h, w, _ = image.shape
center = (w//2, h//2)
angle = float(input("Enter the angle of rotation (in degrees) : "))
M = cv.getRotationMatrix2D(center, angle, 1.0)
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow('Original Image', image)
cv.imshow('Rotated Image', rotated)
cv.waitKey(0)
cv.destroyAllWindows()