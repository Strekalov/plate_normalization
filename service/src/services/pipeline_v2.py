import typing as tp

import numpy as np

from src.services.letters_detector import LettersDetector
from src.services.plate_normalizer_v2 import PlateNormalizer_v2

class SecondPipeline:
    def __init__(
        self,
        letters_detector: LettersDetector,
        plate_normalizer: PlateNormalizer_v2,
    ):
        """
        Инициализирует экземпляр класса SecondPipeline, загружая детектор букв и нормализатор номерных знаков.

        Args:
            letters_detector (LettersDetector): Модель для детекции букв.
            plate_normalizer (PlateNormalizer_v2): Модель для нормализации номерных знаков.
        """
        self._letters_detector = letters_detector
        self._plate_normalizer = plate_normalizer

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание нормализованного номерного знака на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Нормализованное изображение номерного знака.
        """
        # Детекция букв на изображении
        letter_bboxes = self._letters_detector.predict(image)
        
        # Нормализация номерного знака с использованием детектированных букв
        normalized_plate = self._plate_normalizer.normalize(image, letter_bboxes)
        
        return normalized_plate
