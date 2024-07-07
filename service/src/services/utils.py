import cv2
import numpy as np
from fastapi import HTTPException
from turbojpeg import TJPF_RGB, TurboJPEG

# Инициализация TurboJPEG для кодирования и декодирования изображений JPEG
jpeg = TurboJPEG()


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Вычисляет евклидово расстояние между двумя точками.

    Args:
        point1 (np.ndarray): Первая точка.
        point2 (np.ndarray): Вторая точка.

    Returns:
        float: Евклидово расстояние между точками.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance


def calculate_iou(box1, box2):
    """
    Вычисляет коэффициент пересечения (IoU) между двумя прямоугольными областями.

    Args:
        box1 (list): Координаты первой области в формате [x_min, y_min, x_max, y_max].
        box2 (list): Координаты второй области в формате [x_min, y_min, x_max, y_max].

    Returns:
        float: Значение IoU между двумя областями.
    """
    intersection_xmin = max(box1[0], box2[0])
    intersection_ymin = max(box1[1], box2[1])
    intersection_xmax = min(box1[2], box2[2])
    intersection_ymax = min(box1[3], box2[3])

    if intersection_xmax > intersection_xmin and intersection_ymax > intersection_ymin:
        intersection_area = (intersection_xmax - intersection_xmin) * (
            intersection_ymax - intersection_ymin
        )
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    else:
        return 0.0


def merge_boxes(boxes: np.ndarray, threshold: float) -> np.ndarray:
    """
    Объединяет пересекающиеся прямоугольные области на основе заданного порога IoU.

    Args:
        boxes (np.ndarray): Список прямоугольных областей в формате [[x_min, y_min, x_max, y_max], ...].
        threshold (float): Пороговое значение IoU для объединения областей.

    Returns:
        np.ndarray: Объединённый список прямоугольных областей.
    """
    used = [False] * len(boxes)

    i = 0
    while i < len(boxes):
        if not used[i]:
            current_box = boxes[i]
            j = i + 1
            while j < len(boxes):
                if not used[j]:
                    if calculate_iou(current_box, boxes[j]) > threshold:
                        # Объединяем боксы
                        current_box = [
                            min(current_box[0], boxes[j][0]),
                            min(current_box[1], boxes[j][1]),
                            max(current_box[2], boxes[j][2]),
                            max(current_box[3], boxes[j][3]),
                        ]
                        # Помечаем бокс j как использованный
                        used[j] = True
                        # Удаляем бокс j из списка
                        boxes = np.delete(boxes, j, axis=0)
                    else:
                        j += 1
                else:
                    j += 1
            # Заменяем текущий бокс на объединённый
            boxes[i] = current_box
        i += 1

    # Возвращаем список объединённых боксов
    return boxes


def decode_jpg(image_data: bytes) -> np.ndarray:
    """
    Декодирует изображение JPEG из байтового массива.

    Args:
        image_data (bytes): Байтовый массив изображения.

    Returns:
        np.ndarray: Декодированное изображение в формате RGB.

    Raises:
        HTTPException: Если изображение недействительно.
    """
    image = jpeg.decode(image_data, pixel_format=TJPF_RGB)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG image")
    return image


def decode_other(image_data: bytes) -> np.ndarray:
    """
    Декодирует изображение из байтового массива в формат, отличный от JPEG.

    Args:
        image_data (bytes): Байтовый массив изображения.

    Returns:
        np.ndarray: Декодированное изображение в формате RGB.

    Raises:
        HTTPException: Если изображение недействительно.
    """
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    # Преобразование из BGR в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def encode_jpg(image_array: np.ndarray) -> bytes:
    """
    Кодирует изображение в формате RGB в JPEG.

    Args:
        image_array (np.ndarray): Массив изображения в формате RGB.

    Returns:
        bytes: Кодированное изображение в формате JPEG.
    """
    return jpeg.encode(image_array)
