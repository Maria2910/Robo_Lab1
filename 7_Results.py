import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Загружаем изображение
image = cv2.imread('Pizza.jpg')

if image is None:
    print("Ошибка: не удалось загрузить изображение!")
    exit()

# Создаем папку для результатов если её нет
output_dir = 'image_processing_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Создана папка для результатов: {output_dir}")

# ВЫПОЛНЕНИЕ ВСЕХ ПРЕДЫДУЩИХ ЗАДАНИЙ
print("Применение фильтров...")

# 1. Медианный фильтр
median_filtered = cv2.medianBlur(image, 5)

# 2. Гауссово размытие
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# 3. Повышение резкости
sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# 4. Выделение границ (Собель)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.magnitude(sobel_x, sobel_y)
edges = cv2.convertScaleAbs(edges)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# 5. Собственный фильтр
custom_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
custom_filtered = cv2.filter2D(image, -1, custom_kernel)

# 6. Комбинирование результатов
combined = cv2.addWeighted(sharpened, 1.8, edges, -0.8, 50)
combined = np.clip(combined, 0, 255)


# ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ ИЗОБРАЖЕНИЙ
def save_all_images(original, median, gaussian, edges, sharpened, custom, combined):
    """Сохраняет все обработанные изображения"""
    try:
        cv2.imwrite(f'{output_dir}/1_original.jpg', original)
        cv2.imwrite(f'{output_dir}/2_median_filtered.jpg', median)
        cv2.imwrite(f'{output_dir}/3_gaussian_blur.jpg', gaussian)
        cv2.imwrite(f'{output_dir}/4_sobel_edges.jpg', edges)
        cv2.imwrite(f'{output_dir}/5_sharpened.jpg', sharpened)
        cv2.imwrite(f'{output_dir}/6_custom_filter.jpg', custom)
        cv2.imwrite(f'{output_dir}/7_combined_result.jpg', combined)
        print("✅ Все изображения успешно сохранены!")
        print(f"📁 Папка с результатами: {output_dir}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении изображений: {e}")


# ФУНКЦИЯ ДЛЯ ОТОБРАЖЕНИЯ ВСЕХ РЕЗУЛЬТАТОВ
def show_all_results(original, median, gaussian, edges, sharpened, custom, combined):
    """Отображает все результаты в виде сетки"""
    plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.93, bottom=0.01, left=0.05, right=0.95)

    # Оригинальное изображение
    plt.subplot(2, 4, 1)
    plt.title('1. Оригинальное\nизображение', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Медианный фильтр
    plt.subplot(2, 4, 2)
    plt.title('2. Медианный\nфильтр', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Гауссово размытие
    plt.subplot(2, 4, 3)
    plt.title('3. Гауссово\nразмытие', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Выделение границ
    plt.subplot(2, 4, 4)
    plt.title('4. Выделение\nграниц (Собель)', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Повышение резкости
    plt.subplot(2, 4, 5)
    plt.title('5. Повышение\nрезкости', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Собственный фильтр
    plt.subplot(2, 4, 6)
    plt.title('6. Собственный\nфильтр', fontsize=11, pad=15)
    plt.imshow(cv2.cvtColor(custom, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Комбинированный результат
    plt.subplot(2, 4, 7)
    plt.title('7. Комбинированный\nрезультат', fontsize=11, pad=15, color='darkred')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Информационная панель
    plt.subplot(2, 4, 8)
    plt.text(0.1, 0.8, 'ВСЕ РЕЗУЛЬТАТЫ\nСОХРАНЕНЫ В:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.6, f'папку:\n{output_dir}', fontsize=10, color='blue')
    plt.text(0.1, 0.4, 'Файлы:', fontsize=10, fontweight='bold')
    plt.text(0.1, 0.35, '1_original.jpg', fontsize=8)
    plt.text(0.1, 0.3, '2_median_filtered.jpg', fontsize=8)
    plt.text(0.1, 0.25, '3_gaussian_blur.jpg', fontsize=8)
    plt.text(0.1, 0.2, '4_sobel_edges.jpg', fontsize=8)
    plt.text(0.1, 0.15, '5_sharpened.jpg', fontsize=8)
    plt.text(0.1, 0.1, '6_custom_filter.jpg', fontsize=8)
    plt.text(0.1, 0.05, '7_combined_result.jpg', fontsize=8)
    plt.axis('off')

    plt.tight_layout()

    # Сохраняем итоговую сетку
    plt.savefig(f'{output_dir}/all_results_grid.png', dpi=300, bbox_inches='tight')
    print("✅ Сетка с результатами сохранена как 'all_results_grid.png'")

    plt.show()


# ВЫЗОВ ФУНКЦИЙ
print("Отображение всех результатов работы...")

# Сохраняем все отдельные изображения
save_all_images(image, median_filtered, blurred, edges, sharpened, custom_filtered, combined)

# Отображаем результаты
show_all_results(image, median_filtered, blurred, edges, sharpened, custom_filtered, combined)

# Дополнительная информация
print("\n" + "=" * 50)
print("СПИСОК СОХРАНЕННЫХ ФАЙЛОВ:")
print("1_original.jpg - Исходное изображение")
print("2_median_filtered.jpg - После медианного фильтра")
print("3_gaussian_blur.jpg - После гауссова размытия")
print("4_sobel_edges.jpg - Границы Собеля")
print("5_sharpened.jpg - Увеличение резкости")
print("6_custom_filter.jpg - Пользовательский фильтр")
print("7_combined_result.jpg - Комбинированный результат")
print("all_results_grid.png - Сетка со всеми результатами")
print("=" * 50)
