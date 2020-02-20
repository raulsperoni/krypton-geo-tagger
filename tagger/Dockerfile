FROM python
MAINTAINER Raul Speroni (raulsperoni@gmail.com)


# Export env settings
ENV TERM=xterm
ENV LANG en_US.UTF-8

#COPY requirements.txt /tmp
#RUN pip install -r /tmp/requirements.txt

RUN pip install pymongo urllib3[secure] unidecode Geohash geopy folium shapely nltk Flask flasgger spacy spacy-lookup
RUN python -m spacy download es

RUN pip install textacy==0.6.2


COPY libs/ /krypton/
WORKDIR /krypton

CMD ["python", "-u", "geoTaggerServer.py"]




