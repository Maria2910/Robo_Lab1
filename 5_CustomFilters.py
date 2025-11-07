import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()


# Функция для применения собственного фильтра
def apply_custom_filter(img, kernel, filter_name):
    """
    Применяет пользовательский фильтр к изображению

    Parameters:
    img - исходное изображение
    kernel - ядро свертки (2D массив)
    filter_name - название фильтра для отображения
    """
    filtered = cv2.filter2D(img, -1, kernel)
    cv2.imshow(filter_name, filtered)
    return filtered


# 1. ФИЛЬТР ПОВЫШЕНИЯ РЕЗКОСТИ
sharpness_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 2. ФИЛЬТР РАЗМЫТИЯ (УСРЕДНЕНИЯ)
blur_kernel = np.array([
    [0.111, 0.111, 0.111],
    [0.111, 0.111, 0.111],
    [0.111, 0.111, 0.111]
])

# 3. ФИЛЬТР ДЛЯ ВЫДЕЛЕНИЯ КРАЕВ (ЛАПЛАСИАН)
edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# 4. ФИЛЬТР ТИСНЕНИЯ (EMBOSS)
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

# 5. ФИЛЬТР ДЛЯ ВЫДЕЛЕНИЯ ГОРИЗОНТАЛЬНЫХ ЛИНИЙ
horizontal_kernel = np.array([
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]
])

# 6. ФИЛЬТР ДЛЯ ВЫДЕЛЕНИЯ ВЕРТИКАЛЬНЫХ ЛИНИЙ
vertical_kernel = np.array([
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]
])

# 7. ГАУССОВО РАЗМЫТИЕ (ПРИБЛИЖЕННОЕ)
gaussian_kernel = (1 / 16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

# Применяем все фильтры
print("Применяем пользовательские фильтры...")

# Отображаем оригинал
cv2.imshow('0. Original Image', image)

# Применяем фильтры
sharpened = apply_custom_filter(image, sharpness_kernel, '1. Sharpness Filter')
blurred = apply_custom_filter(image, blur_kernel, '2. Blur Filter')
edges = apply_custom_filter(image, edge_detection_kernel, '3. Edge Detection')
emboss = apply_custom_filter(image, emboss_kernel, '4. Emboss Filter')
horizontal = apply_custom_filter(image, horizontal_kernel, '5. Horizontal Lines')
vertical = apply_custom_filter(image, vertical_kernel, '6. Vertical Lines')
gaussian = apply_custom_filter(image, gaussian_kernel, '7. Gaussian Blur')

print("\nВсе фильтры применены!")
print("Нажмите любую клавишу для закрытия окон...")

cv2.waitKey(0)
cv2.destroyAllWindows()
