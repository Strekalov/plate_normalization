import typing as tp

import numpy as np
from ultralytics import YOLO

from src.services.utils import merge_boxes

class LettersDetector:
    def __init__(self, config: tp.Dict):
        """
        Инициализирует экземпляр класса LettersDetector, загружая конфигурацию и модель.

        Args:
            config (tp.Dict): Словарь конфигурации с путём к модели, размером изображения и параметрами уверенности.
        """
        self._model_path = config["model_path"]
        self._imgsz = config["image_size"]
        self._conf = config["conf"]
        self._model = YOLO(self._model_path, task="detect")

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание букв на изображении и постобработку результата.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Список координат боксов букв.
        """
        return self._postprocess_predict(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание букв на изображении с использованием модели.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Предсказанные боксы букв.
        """
        return self._model(image, conf=self._conf, device="cuda:0")

    def _postprocess_predict(self, predicts: list) -> tp.List[str]:
        """
        Постобработка предсказанных боксов для получения сортированных координат.

        Args:
            predicts (list): Список предсказаний боксов.

        Returns:
            tp.List[str]: Сортированные координаты боксов букв.
        """
        bboxes = np.array([r.boxes.xyxy.cpu().numpy() for r in predicts])[0]

        # Объединение перекрывающихся боксов
        bboxes = merge_boxes(bboxes, threshold=0.7)

        # Сортировка боксов по координате y
        bboxes = sorted(bboxes, key=lambda x: x[1])

        two_rows_flag = False

        if len(bboxes) >= 6:
            bbox_0 = bboxes[0]
            bbox_4 = bboxes[4]
            bbox_5 = bboxes[5]

            # Нижняя граница 0-го бокса и его высота
            bottom_0 = bbox_0[3]
            height_0 = bbox_0[3] - bbox_0[1]

            # Верхняя граница и нижняя граница 4-го бокса
            top_4 = bbox_4[1]
            bottom_4 = bbox_4[3]

            # Высота 5-го бокса
            height_5 = bbox_5[3] - bbox_5[1]
            top_5 = bbox_5[1]

            # Проверка условий и установка флага для двух рядов
            if (
                top_4 >= bottom_0 - 0.1 * height_0
                and top_5 <= bottom_4 - 0.5 * height_5
            ):
                two_rows_flag = True

        if two_rows_flag:
            # Сортировка боксов в два ряда
            bboxes_1 = sorted(bboxes[:4], key=lambda x: x[0])
            bboxes_2 = sorted(bboxes[4:], key=lambda x: x[0])
            bboxes = bboxes_1 + bboxes_2
        else:
            # Сортировка боксов в один ряд
            bboxes = sorted(bboxes, key=lambda x: x[0])

        sort_bboxes = bboxes

        return sort_bboxes
