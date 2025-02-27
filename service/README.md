<h1 align="center">API Service для выравнивания и распознавания автомобильных номеров</h1>



## Порядок работы с проектом

### Прежде, чем начать

установите docker, nvidia-docker, nvidia-driver-545+

### 1. настройка окружения

Для сборка docker образа введите

```bash
make build_docker
```

### 2. Запуск сервиса

Соберите движки TensorRT

```bash
make build_trt_engine
```

Измените config/config.yaml для указания корректных путей к моделям (не обязательно tensorrt, можно и pt) 
Так же можно поменять параметры API сервиса такие как порт и воркер в Makefile


Запустите сервис
```bash
make run_docker_app
```

### 3. Нагрузочное тестирование

В проекте реализовано нагрузочное тестирование с помощью фреймворка locust


Запустите тестирование
```bash
make locust_test
```
