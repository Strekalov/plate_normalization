import typing as tp

import numpy as np
from ultralytics import YOLO

from src.services.utils import euclidean_distance


class KeypointsDetector:
    def __init__(self, config: tp.Dict):
        """
        Инициализирует экземпляр класса KeypointsDetector, загружая конфигурацию и модель.

        Args:
            config (tp.Dict): Словарь конфигурации с путём к модели, размером изображения и параметрами уверенности.
        """
        self._model_path = config["model_path"]
        self._imgsz = config["image_size"]
        self._conf = config["conf"]
        self._model = YOLO(self._model_path, task="pose")

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание ключевых точек на изображении и постобработку результата.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Список координат ключевых точек.
        """
        return self._postprocess_predict(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание ключевых точек на изображении с использованием модели.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Предсказанные ключевые точки.
        """
        return self._model(image, conf=self._conf, device="cuda:0")

    def _get_largest_plate_by_keypoints(self, predicts: list) -> np.ndarray:
        """
        Находит объект с наибольшей суммой расстояний между ключевыми точками.

        Args:
            predicts (list): Список предсказаний ключевых точек.

        Returns:
            np.ndarray: Ключевые точки объекта с наибольшей суммой расстояний.
        """
        max_distance_sum = 0
        best_keypoints = None
        for p in predicts:
            keypoints = p.keypoints.xy[0].cpu().numpy()

            distance_0_1 = euclidean_distance(keypoints[0], keypoints[1])
            distance_0_2 = euclidean_distance(keypoints[0], keypoints[2])
            distance_sum = distance_0_1 + distance_0_2

            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                best_keypoints = keypoints

        return best_keypoints

    def _postprocess_predict(self, predicts: list) -> tp.List[str]:
        """
        Постобработка предсказанных ключевых точек для получения наибольшего объекта.

        Args:
            predicts (list): Список предсказаний ключевых точек.

        Returns:
            tp.List[str]: Ключевые точки наибольшего объекта.
        """
        return self._get_largest_plate_by_keypoints(predicts)
