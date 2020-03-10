FROM python
MAINTAINER Raul Speroni (raulsperoni@gmail.com)

ENV TERM=xterm
ENV LANG en_US.UTF-8

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir /krypton
COPY . /krypton/
WORKDIR /krypton

CMD ["python", "-u", "geoTaggerServer.py"]




