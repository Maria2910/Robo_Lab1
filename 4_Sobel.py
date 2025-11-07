import cv2

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()

# Конвертируем в grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ПРИМЕНЕНИЕ ОПЕРАТОРА СОБЕЛЯ

# Градиенты с сохранением знака
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Комбинирование градиентов
edges_combined = cv2.magnitude(sobel_x, sobel_y)
edges_combined = cv2.convertScaleAbs(edges_combined)

# Конвертируем отдельные градиенты для отображения
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Отображаем результаты
cv2.imshow('1. Original Image', image)
cv2.imshow('2. Sobel X (Vertical Edges)', sobel_x_abs)
cv2.imshow('3. Sobel Y (Horizontal Edges)', sobel_y_abs)
cv2.imshow('4. Combined Edges (magnitude)', edges_combined)

print("Оператор Собеля:")
print("2. Sobel X - вертикальные границы")
print("3. Sobel Y - горизонтальные границы")
print("4. Combined - все границы")

cv2.waitKey(0)
cv2.destroyAllWindows()
