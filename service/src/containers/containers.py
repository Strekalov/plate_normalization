from dependency_injector import containers, providers

from src.services.beeline_ocr import BeelineOCR
from src.services.keypoints_detector import KeypointsDetector
from src.services.letters_detector import LettersDetector
from src.services.ocr_pipeline import OCRPipeline
from src.services.pipeline_v1 import FirstPipeline
from src.services.pipeline_v2 import SecondPipeline
from src.services.plate_normalizer import PlateNormalizer
from src.services.plate_normalizer_v2 import PlateNormalizer_v2


class AppContainer(containers.DeclarativeContainer):
    """
    Контейнер для зависимостей приложения, использующий dependency_injector.
    """

    config = providers.Configuration()

    keypoints_detector = providers.Singleton(
        KeypointsDetector,
        config=config.services.keypoints_detector,
    )

    letters_detector = providers.Singleton(
        LettersDetector,
        config=config.services.letters_detector,
    )

    plate_normalizer = providers.Singleton(
        PlateNormalizer,
    )

    plate_normalizer_v2 = providers.Singleton(
        PlateNormalizer_v2,
        config=config.services.plate_normalizer_v2,
    )

    beeline_ocr = providers.Singleton(
        BeelineOCR,
        config=config.services.beeline_ocr,
    )

    first_pipeline = providers.Singleton(
        FirstPipeline,
        keypoints_detector=keypoints_detector,
        plate_normalizer=plate_normalizer,
    )

    second_pipeline = providers.Singleton(
        SecondPipeline,
        letters_detector=letters_detector,
        plate_normalizer=plate_normalizer_v2,
    )

    ocr_pipeline_v1 = providers.Singleton(
        OCRPipeline,
        normalization_pipeline=first_pipeline,
        ocr=beeline_ocr,
    )

    ocr_pipeline_v2 = providers.Singleton(
        OCRPipeline,
        normalization_pipeline=second_pipeline,
        ocr=beeline_ocr,
    )
