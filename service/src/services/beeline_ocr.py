import json
import typing as tp

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

class BeelineOCR:
    def __init__(self, config: tp.Dict):
        """
        Инициализирует экземпляр класса BeelineOCR, загружая конфигурацию и модели.

        Args:
            config (tp.Dict): Словарь конфигурации с путями к моделям и шаблонам.
        """
        self._model_path = config["model_path"]
        self._imgsz = config["image_size"]
        self._template_two_digits_path = config["template_two_digits_path"]
        self._template_three_digits_path = config["template_three_digits_path"]

        with open(self._template_two_digits_path, "r") as file:
            self._template_two_digits = json.load(file)
        with open(self._template_three_digits_path, "r") as file:
            self._template_three_digits = json.load(file)

        self._model = self._build_model()
        self.label2letter = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "A",
            11: "B",
            12: "C",
            13: "E",
            14: "H",
            15: "K",
            16: "M",
            17: "O",
            18: "P",
            19: "T",
            20: "X",
            21: "Y",
        }

    def _build_model(self) -> torch.nn.Module:
        """
        Создаёт и загружает модель ResNet18 для OCR.

        Returns:
            torch.nn.Module: Загруженная и настроенная модель ResNet18.
        """
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, out_features=22)
        model.load_state_dict(
            torch.load(self._model_path, map_location=torch.device("cpu"))
        )
        model.eval()
        model.to(device="cuda:0")
        return model

    def _transform(self, image: np.ndarray) -> torch.Tensor:
        """
        Препроцессинг изображения для модели OCR.

        Args:
            image (np.ndarray): Исходное изображение.

        Returns:
            torch.Tensor: Изображение, преобразованное в тензор Torch.
        """
        # Изменение размера изображения с использованием OpenCV
        image = cv2.resize(image, (self._imgsz, self._imgsz))

        # Преобразование изображения в тензор и нормализация
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # Изменение порядка осей на CxHxW

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image = (image - mean) / std
        return image.to(device="cuda:0")

    def _get_crops_from_template(
        self, image: np.ndarray, region_length: int
    ) -> tp.List[np.ndarray]:
        """
        Получает кропы из исходного изображения по шаблонам.

        Args:
            image (np.ndarray): Исходное изображение.
            region_length (int): Количество символов в регионе (2 или 3).

        Raises:
            ValueError: Если задано недопустимое количество символов в регионе.

        Returns:
            tp.List[np.ndarray]: Список кропов изображения по шаблонам.
        """
        if region_length == 2:
            pattern = self._template_two_digits
        elif region_length == 3:
            pattern = self._template_three_digits
        else:
            raise ValueError(
                "Неподдерживаемое разбиение на регионы. Поддерживаются только 2 и 3."
            )

        H, W, _ = image.shape

        crops = []
        for pos in pattern:
            sx, sy, ex, ey = *pos["p1"], *pos["p2"]
            sx, sy, ex, ey = sx * W, sy * H, ex * W, ey * H
            sx, sy, ex, ey = map(int, [sx, sy, ex, ey])
            crops.append(image[sy:ey, sx:ex])

        return crops

    def _predict_one_symbol(self, image: np.ndarray) -> str:
        """
        Предсказывает один символ с помощью модели OCR.

        Args:
            image (np.ndarray): Один кроп, вырезанный по шаблону.

        Returns:
            str: Предсказанный символ.
        """
        input_tensor = self._transform(image)
        output_tensor = self._model(input_tensor.unsqueeze(0))
        predicted = torch.argmax(output_tensor)
        return self.label2letter[predicted.item()]

    def _predict_series(self, images: tp.List[np.ndarray]) -> str:
        """
        Предсказывает символ на каждом кропе по шаблону.

        Args:
            images (tp.List[np.ndarray]): Список кропов по шаблону.

        Returns:
            str: Предсказанный номер.
        """
        plate_text = ""
        for image in images:
            plate_text += self._predict_one_symbol(image)
        return plate_text

    def _postprocess(self, plate_text: str) -> str:
        """
        Постобработка предсказанного текста.
        Если на месте буквы предсказана цифра или наоборот,
        заменяет на корректный символ из словаря.

        Args:
            plate_text (str): Предсказанный текст номера.

        Returns:
            str: Постобработанный текст номера.
        """
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        letters_en = ["E", "T", "Y", "O", "P", "A", "H", "K", "X", "C", "B", "M"]
        from_digits_to_letters = {
            "1": "T",
            "2": "A",
            "3": "B",
            "4": "H",
            "5": "B",
            "6": "C",
            "7": "T",
            "8": "B",
            "9": "Y",
            "0": "O",
        }
        from_letters_to_digits = {
            "A": "4",
            "B": "8",
            "E": "8",
            "K": "4",
            "M": "3",
            "H": "4",
            "O": "0",
            "P": "9",
            "C": "6",
            "T": "7",
            "Y": "9",
            "X": "8",
        }

        new_number = ""
        if plate_text[0] in digits:
            new_number += from_digits_to_letters[plate_text[0]]
        else:
            new_number += plate_text[0]
        for i in range(1, 4):
            if plate_text[i] in letters_en:
                new_number += from_letters_to_digits[plate_text[i]]
            else:
                new_number += plate_text[i]
        for i in range(4, 6):
            if plate_text[i] in digits:
                new_number += from_digits_to_letters[plate_text[i]]
            else:
                new_number += plate_text[i]
        for i in range(6, len(plate_text)):
            if plate_text[i] in letters_en:
                new_number += from_letters_to_digits[plate_text[i]]
            else:
                new_number += plate_text[i]
        return new_number

    def predict(self, image: np.ndarray) -> tp.Tuple[str, str]:
        """
        Предсказывает номер по изображению, возвращая два варианта с разными шаблонами.

        Args:
            image (np.ndarray): Исходное изображение.

        Returns:
            tp.Tuple[str, str]: Два варианта предсказанного номера.
        """
        result_ocr_r2 = self._predict_series(self._get_crops_from_template(image, 2))
        result_ocr_r3 = self._predict_series(self._get_crops_from_template(image, 3))
        return self._postprocess(result_ocr_r2), self._postprocess(result_ocr_r3)
