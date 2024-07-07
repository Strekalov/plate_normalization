FROM nvcr.io/nvidia/pytorch:24.06-py3

RUN apt-get update && apt-get install libturbojpeg

WORKDIR /plate_normalization_service

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . /plate_normalization_service/

EXPOSE 8888

CMD make run_app




