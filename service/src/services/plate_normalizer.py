import cv2
import numpy as np


class PlateNormalizer:
    def normalize(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Нормализует изображение номерного знака с использованием координат четырёх точек.

        Args:
            image (np.ndarray): Входное изображение.
            pts (np.ndarray): Координаты четырёх точек для трансформации.

        Returns:
            np.ndarray: Нормализованное изображение.
        """
        return self._four_point_transform(image, pts)

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Выполняет перспективную трансформацию изображения с использованием координат четырёх точек.

        Args:
            image (np.ndarray): Входное изображение.
            pts (np.ndarray): Координаты четырёх точек для трансформации.

        Returns:
            np.ndarray: Трансформированное изображение.
        """
        (tl, tr, br, bl) = pts

        # Вычисление ширины результирующего изображения
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Вычисление высоты результирующего изображения
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Координаты целевого изображения
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        # Вычисление матрицы перспективного преобразования
        M = cv2.getPerspectiveTransform(pts, dst)

        # Применение перспективного преобразования к изображению
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped
