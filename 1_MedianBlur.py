import cv2

cap = cv2.VideoCapture(0)  # Открываем веб-камеру

while True:
    ret, frame = cap.read()  # Читаем кадр с камеры

    # ПРИМЕНЕНИЕ МЕДИАННОГО ФИЛЬТРА
    filtered_frame = cv2.medianBlur(frame, 5)  # Размер ядра = 5x5

    # Отображаем оба кадра для сравнения
    cv2.imshow('Original', frame)
    cv2.imshow('Filtered', filtered_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
