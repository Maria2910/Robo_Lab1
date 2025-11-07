import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

# Проверяем, загрузилось ли изображение
if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()

# ПРИМЕНЕНИЕ ГАУССОВА РАЗМЫТИЯ
# Параметры: (изображение, размер_ядра, стандартное_отклонение)
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Отображаем результаты
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blurred', blurred_image)

print("Нажмите любую клавишу для закрытия окон...")
cv2.waitKey(0)
cv2.destroyAllWindows()
