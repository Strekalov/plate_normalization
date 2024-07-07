import time
from io import BytesIO

import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from fastapi.responses import StreamingResponse

from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.pipeline_v1 import FirstPipeline
from src.services.pipeline_v2 import SecondPipeline
from src.services.utils import decode_jpg, decode_other, encode_jpg


@router.post("/get_normalized_image_v1")
@inject
def get_normalized_image_v1(
    image: bytes = File(),
    service: FirstPipeline = Depends(Provide[AppContainer.first_pipeline]),
):
    """
    Обрабатывает запрос на получение нормализованного изображения версии 1.

    Args:
        image (bytes): Входное изображение.
        service (FirstPipeline): Сервис для обработки изображения.

    Returns:
        StreamingResponse: Нормализованное изображение в формате JPEG.
    """
    t1 = time.time()
    # Проверка на формат JPEG
    if image[:3] == b"\xff\xd8\xff":
        img: np.ndarray = decode_jpg(image)
    else:
        img: np.ndarray = decode_other(image)

    normalized_image = service.predict(img)

    # Кодирование изображения в формат JPEG
    jpeg_bytes = encode_jpg(normalized_image)
    headers = {
        "processing_time": f"{time.time() - t1}",
    }
    # Возврат изображения как HTTP-ответ
    return StreamingResponse(
        BytesIO(jpeg_bytes), media_type="image/jpeg", headers=headers
    )


@router.post("/get_normalized_image_v2")
@inject
def get_normalized_image_v2(
    image: bytes = File(),
    service: SecondPipeline = Depends(Provide[AppContainer.second_pipeline]),
):
    """
    Обрабатывает запрос на получение нормализованного изображения версии 2.

    Args:
        image (bytes): Входное изображение.
        service (SecondPipeline): Сервис для обработки изображения.

    Returns:
        StreamingResponse: Нормализованное изображение в формате JPEG.
    """
    t1 = time.time()
    # Проверка на формат JPEG
    if image[:3] == b"\xff\xd8\xff":
        img: np.ndarray = decode_jpg(image)
    else:
        img: np.ndarray = decode_other(image)

    normalized_image = service.predict(img)

    # Кодирование изображения в формат JPEG
    jpeg_bytes = encode_jpg(normalized_image)

    headers = {
        "processing_time": f"{time.time() - t1}",
    }
    # Возврат изображения как HTTP-ответ
    return StreamingResponse(
        BytesIO(jpeg_bytes), media_type="image/jpeg", headers=headers
    )


@router.post("/get_normalized_image_v1_ocr")
@inject
def get_normalized_image_v1_ocr(
    image: bytes = File(),
    service: SecondPipeline = Depends(Provide[AppContainer.ocr_pipeline_v1]),
):
    """
    Обрабатывает запрос на получение нормализованного изображения версии 1 с OCR.

    Args:
        image (bytes): Входное изображение.
        service (SecondPipeline): Сервис для обработки изображения и OCR.

    Returns:
        StreamingResponse: Нормализованное изображение в формате JPEG и предсказанный текст.
    """
    t1 = time.time()
    # Проверка на формат JPEG
    if image[:3] == b"\xff\xd8\xff":
        img: np.ndarray = decode_jpg(image)
    else:
        img: np.ndarray = decode_other(image)

    result = service.predict(img)

    # Кодирование изображения в формат JPEG
    jpeg_bytes = encode_jpg(result.normalized_plate)

    headers = {
        "plate_text_r2": result.plate_text_r2,
        "plate_text_r3": result.plate_text_r3,
        "processing_time": f"{time.time() - t1}",
    }

    # Возврат изображения как HTTP-ответ
    return StreamingResponse(
        BytesIO(jpeg_bytes), media_type="image/jpeg", headers=headers
    )


@router.post("/get_normalized_image_v2_ocr")
@inject
def get_normalized_image_v2_ocr(
    image: bytes = File(),
    service: SecondPipeline = Depends(Provide[AppContainer.ocr_pipeline_v2]),
):
    """
    Обрабатывает запрос на получение нормализованного изображения версии 2 с OCR.

    Args:
        image (bytes): Входное изображение.
        service (SecondPipeline): Сервис для обработки изображения и OCR.

    Returns:
        StreamingResponse: Нормализованное изображение в формате JPEG и предсказанный текст.
    """
    t1 = time.time()
    # Проверка на формат JPEG
    if image[:3] == b"\xff\xd8\xff":
        img: np.ndarray = decode_jpg(image)
    else:
        img: np.ndarray = decode_other(image)

    result = service.predict(img)

    # Кодирование изображения в формат JPEG
    jpeg_bytes = encode_jpg(result.normalized_plate)

    headers = {
        "plate_text_r2": result.plate_text_r2,
        "plate_text_r3": result.plate_text_r3,
        "processing_time": f"{time.time() - t1}",
    }
    # Возврат изображения как HTTP-ответ
    return StreamingResponse(
        BytesIO(jpeg_bytes), media_type="image/jpeg", headers=headers
    )
