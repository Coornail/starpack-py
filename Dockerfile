FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app

COPY ./* ./
RUN pip install --no-cache-dir -r requirements.txt

VOLUME ROOT

CMD python3 ./starpack.py
