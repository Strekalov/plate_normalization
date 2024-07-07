import json
import typing as tp

import cv2
import numpy as np


class PlateNormalizer_v2:
    def __init__(self, config: tp.Dict):
        """
        Инициализирует экземпляр класса PlateNormalizer_v2, загружая шаблоны для нормализации номерных знаков.

        Args:
            config (tp.Dict): Словарь конфигурации с путями к шаблонам для двух и трех значных номеров.
        """
        self._template_two_digits_path = config["template_two_digits_path"]
        self._template_three_digits_path = config["template_three_digits_path"]

        with open(self._template_two_digits_path, "r") as file:
            self._template_two_digits = json.load(file)
        with open(self._template_three_digits_path, "r") as file:
            self._template_three_digits = json.load(file)

    def normalize(self, image: np.ndarray, letter_boxes: np.ndarray) -> np.ndarray:
        """
        Нормализует изображение номерного знака с использованием координат букв.

        Args:
            image (np.ndarray): Входное изображение.
            letter_boxes (np.ndarray): Координаты боксов букв.

        Returns:
            np.ndarray: Нормализованное изображение.
        """
        return self._normalize(image, letter_boxes)

    def _normalize(self, image: np.ndarray, letter_boxes: np.ndarray) -> np.ndarray:
        """
        Выполняет нормализацию изображения номерного знака.

        Args:
            image (np.ndarray): Входное изображение.
            letter_boxes (np.ndarray): Координаты боксов букв.

        Returns:
            np.ndarray: Нормализованное изображение.
        """
        letter_crops = self._get_letter_crops(image, letter_boxes)
        normalized_image = self._make_normalized_image(image, letter_crops)
        return normalized_image

    def _get_letter_crops(
        self, image: np.ndarray, letter_boxes: np.ndarray
    ) -> tp.List[np.ndarray]:
        """
        Извлекает кропы букв из изображения по координатам боксов.

        Args:
            image (np.ndarray): Входное изображение.
            letter_boxes (np.ndarray): Координаты боксов букв.

        Returns:
            tp.List[np.ndarray]: Список кропов букв.
        """
        cropped_images = []
        for box in letter_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped_image)
        return cropped_images

    def _make_normalized_image(
        self, image: np.ndarray, cropped_images: tp.List[np.ndarray]
    ) -> np.ndarray:
        """
        Создает нормализованное изображение номерного знака из кропов букв.

        Args:
            image (np.ndarray): Входное изображение.
            cropped_images (tp.List[np.ndarray]): Список кропов букв.

        Returns:
            np.ndarray: Нормализованное изображение.
        """
        h, w, _ = image.shape
        # Создание пустого холста
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image.fill(255)

        new_height, new_width, _ = new_image.shape

        for id, cropped_image in enumerate(cropped_images, 1):
            pos = id
            template = (
                self._template_two_digits
                if len(cropped_images) <= 8
                else self._template_three_digits
            )

            p1, p2 = self._get_p1_p2(pos, template)

            if p1 is None or p2 is None:
                print(f"Error: Position {pos} not found in the template")
                continue

            x1 = int(p1[0] * new_width)
            y1 = int(p1[1] * new_height)
            x2 = int(p2[0] * new_width)
            y2 = int(p2[1] * new_height)

            resized_cropped_image = cv2.resize(cropped_image, (x2 - x1, y2 - y1))
            new_image[y1:y2, x1:x2] = resized_cropped_image

        return new_image

    def _get_p1_p2(
        self, pos: int, template: list
    ) -> tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]]:
        """
        Получает координаты p1 и p2 из шаблона для заданной позиции.

        Args:
            pos (int): Позиция символа.
            template (list): Шаблон с координатами.

        Returns:
            tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]]: Координаты p1 и p2.
        """
        for item in template:
            if item["pos"] == pos:
                return item["p1"], item["p2"]
        return None, None
