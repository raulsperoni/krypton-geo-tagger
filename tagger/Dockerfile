FROM python:3.4-slim
MAINTAINER Raul Speroni (raulsperoni@gmail.com)

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y\
    libgeos-dev


# Export env settings
ENV TERM=xterm
ENV LANG en_US.UTF-8

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY libs/ /krypton/
WORKDIR /krypton


 CMD ["python", "-u", "geoTaggerServer.py"]




