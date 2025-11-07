import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()

# ПОДГОТОВКА БАЗОВЫХ КОМПОНЕНТОВ
blurred = cv2.GaussianBlur(image, (15, 15), 0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.magnitude(sobel_x, sobel_y)
edges = cv2.convertScaleAbs(edges)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

print("Создание комбинированных эффектов...")

# 1. ЭФФЕКТ "ПОСТЕРНЫЙ" - яркие цвета, черные контуры
poster_effect = cv2.addWeighted(sharpened, 1.8, edges, -0.8, 50)
poster_effect = np.clip(poster_effect, 0, 255)  # Обрезаем значения

# 2. ЭФФЕКТ "ПРИЗРАЧНЫЙ" - блеклые цвета, белые контуры
ghost_edges = 255 - edges  # Инвертируем границы
ghost_effect = cv2.addWeighted(blurred, 0.4, ghost_edges, 0.6, 80)
ghost_effect = cv2.addWeighted(ghost_effect, 0.3, image, 0.2, 0)

# 3. ЭФФЕКТ "ГРАФИЧЕСКИЙ Ч/Б" - только структура без цветов
gray_structure = cv2.addWeighted(edges, 0.7, blurred, 0.3, 0)
gray_structure = cv2.cvtColor(gray_structure, cv2.COLOR_BGR2GRAY)
graphic_bw = cv2.cvtColor(gray_structure, cv2.COLOR_GRAY2BGR)

# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
cv2.imshow('Original', image)
cv2.imshow('1. POSTER', poster_effect)
cv2.imshow('2. GHOST', ghost_effect)
cv2.imshow('3. GRAPHIC', graphic_bw)

print("1. ПОСТЕР - гиперконтрастный с черными контурами")
print("2. ПРИЗРАК - бледный с белыми контурами")
print("3. ГРАФИКА - структурный черно-белый")
print("Нажмите любую клавишу для закрытия окон...")

cv2.waitKey(0)
cv2.destroyAllWindows()
