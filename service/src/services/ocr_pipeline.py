from dataclasses import dataclass

import numpy as np

from src.services.beeline_ocr import BeelineOCR
from src.services.pipeline_v1 import FirstPipeline
from src.services.pipeline_v2 import SecondPipeline

@dataclass
class OCRResult:
    normalized_plate: np.ndarray
    plate_text_r2: str
    plate_text_r3: str

class OCRPipeline:
    def __init__(
        self,
        normalization_pipeline: FirstPipeline | SecondPipeline,
        ocr: BeelineOCR,
    ):
        """
        Инициализирует экземпляр класса OCRPipeline, загружая пайплайн нормализации и OCR модель.

        Args:
            normalization_pipeline (FirstPipeline | SecondPipeline): Пайплайн для нормализации изображений.
            ocr (BeelineOCR): Модель OCR для распознавания текста.
        """
        self._normalization_pipeline = normalization_pipeline
        self._ocr = ocr

    def predict(self, image: np.ndarray) -> OCRResult:
        """
        Выполняет предсказание текста на изображении, используя пайплайн нормализации и модель OCR.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            OCRResult: Результат OCR, включающий нормализованное изображение и распознанный текст.
        """
        # Нормализуем изображение с использованием пайплайна нормализации
        normalized_plate = self._normalization_pipeline.predict(image)

        # Выполняем OCR для распознавания текста на нормализованном изображении
        plate_text_r2, plate_text_r3 = self._ocr.predict(normalized_plate)

        # Возвращаем результат OCR
        return OCRResult(
            normalized_plate=normalized_plate,
            plate_text_r2=plate_text_r2,
            plate_text_r3=plate_text_r3,
        )
