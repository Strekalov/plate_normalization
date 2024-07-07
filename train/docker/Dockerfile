FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update -y && apt install --no-install-recommends libturbojpeg

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /workspace/project
CMD [ "/bin/bash" ]
