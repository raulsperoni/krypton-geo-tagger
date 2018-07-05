#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import re
import unicodedata



def getTokensNoUserNoHashtag(text):
    """
    Saca tildes, Saca espacios extras
    """
    from unidecode import unidecode
    import re
    import string
    #Saco urls, dejo espacio
    #text = strip_accents(text)
    text = text.strip()
    text = re.sub('http(.)*',' ',text)
    #Saco todo lo que no sea palabra, dejo espacio
    text = re.sub('[^#a-zA-Z0-9_]',' ',text) 
    #Busco hashtag
    hashtags = re.findall(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", text)
    new_words = []
    for hashtag in hashtags:
        hashtag = re.sub('#', '', hashtag)
        for word in camel_case_split(hashtag):
            new_words.append(word)
    text += ' '.join(new_words)
    text = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",' ',text) 
    #Saco espacios de mas
    text = re.sub(' +',' ',text)
    #text = unidecode(text)
    text = text.strip()
    return text.split(' ')



def getTerminosComunesDominioDict():
    """
    """
    result = {}
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, 'palabras_comunes_dominio.txt')
    abvs = open(filename).read().splitlines()
    for abv in abvs:
        result[strip_accents(abv.lower())] = True
    return result



def getStopWordDict():
    """
    """
    result = {}
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, 'stopwords.txt')
    stopwords = open(filename,encoding='utf-8').read().splitlines()
    for stopword in stopwords:
        result[strip_accents(stopword.lower())] = True
    return result

def getTerminosBusquedaDict():
    """
    """
    result = {}
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, 'palabras_clave_ciudad.txt')
    stopwords = open(filename,encoding='utf-8').read().splitlines()
    filename = os.path.join(here, 'palabras_clave_dominio.txt')
    stopwords += open(filename,encoding='utf-8').read().splitlines()
    texto_armado = ' '.join(stopwords)
    tokens = getTokensNoUserNoHashtag(texto_armado)
    for token in tokens:
        result[token.lower()] = True
    return result


    

def getCityBoundingBox():
    """
    """
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, 'city_bounding_box.txt')
    bounding_box = open(filename,encoding='utf-8').read()
    result = []
    for coord in bounding_box.split(','):
        result.append(float(coord))
    return result


def strip_accents(text):
    import re
    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)
    
