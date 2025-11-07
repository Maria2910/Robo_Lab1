import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()

# Метод 3.1: Повышение резкости с помощью свертки
def sharpen_convolution(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

# Метод 3.2: Маска нерезкости (Unsharp Mask)
def unsharp_mask(img, kernel_size=(5, 5), alpha=1.5, beta=-0.5, gamma=0):
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, gamma)
    return sharpened

# Применяем оба метода
sharpened_conv = sharpen_convolution(image)
sharpened_unsharp = unsharp_mask(image)

# Отображаем в трех отдельных окнах
cv2.imshow('1. Original Image', image)
cv2.imshow('2. Sharpened - Convolution Method', sharpened_conv)
cv2.imshow('3. Sharpened - Unsharp Mask Method', sharpened_unsharp)

print("Нажмите любую клавишу для закрытия окон...")

cv2.waitKey(0)
cv2.destroyAllWindows()
