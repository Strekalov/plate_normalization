from roboflow import Roboflow
from dotenv import load_dotenv
import os

# Загружаем переменные из .env файла
load_dotenv()

# Получаем значения переменных
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=roboflow_api_key)

'''
Скачивание основных датасетов для yolo детектора
'''
project = rf.workspace("ru-anrp").project("russian-license-plate-characters-detector")
version = project.version(6)
dataset = version.download("yolov8", location='datasets/russian-license-plate-characters-detector')

project = rf.workspace("project-pq69h").project("mtuci-vkr2")
version = project.version(10)
dataset = version.download("yolov8", location='datasets/mtuci-vkr2')

project = rf.workspace("kolzek").project("russian-car-plate-letters")
version = project.version(2)
dataset = version.download("yolov8", location='datasets/russian-car-plate-letters')

project = rf.workspace("collage-s2ncm").project("plates-3tori")
version = project.version(5)
dataset = version.download("yolov8", location='datasets/plates-3tori')

project = rf.workspace("zalessie").project("plate-9bie5")
version = project.version(1)
dataset = version.download("yolov8", location='datasets/plate-9bie5')

project = rf.workspace("workspace-9qpga").project("car-plates-text")
version = project.version(4)
dataset = version.download("yolov8", location='datasets/car-plates-text')

project = rf.workspace("colab-colab-imaca").project("plates-vlquf")
version = project.version(5)
dataset = version.download("yolov8", location='datasets/plates_keypoints')