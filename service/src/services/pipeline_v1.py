import typing as tp

import numpy as np

from src.services.keypoints_detector import KeypointsDetector
from src.services.plate_normalizer import PlateNormalizer

class FirstPipeline:
    def __init__(
        self,
        keypoints_detector: KeypointsDetector,
        plate_normalizer: PlateNormalizer,
    ):
        """
        Инициализирует экземпляр класса FirstPipeline, загружая детектор ключевых точек и нормализатор номера.

        Args:
            keypoints_detector (KeypointsDetector): Модель для детекции ключевых точек.
            plate_normalizer (PlateNormalizer): Модель для нормализации номерных знаков.
        """
        self._keypoints_detector = keypoints_detector
        self._plate_normalizer = plate_normalizer

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание нормализованного номерного знака на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Нормализованное изображение номерного знака.
        """
        # Детекция ключевых точек на изображении
        keypoints = self._keypoints_detector.predict(image)
        
        # Нормализация номерного знака с использованием ключевых точек
        normalized_plate = self._plate_normalizer.normalize(image, keypoints)
        
        return normalized_plate
