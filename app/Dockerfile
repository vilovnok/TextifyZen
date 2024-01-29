FROM python:3.10-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

RUN apt-get update && \
    apt install -y python3-dev
RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip3 install -r requirements.txt

EXPOSE 8000
COPY . .