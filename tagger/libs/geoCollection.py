#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk import ngrams
from pymongo import MongoClient
from shapely.geometry import asShape
import pymongo
import utils
import logging
logger = logging.getLogger(__name__)

class GeoCollection(object):
    """
    Objeto generico del espacio.
    """
    def __init__(self,mongostring,collection):
        try:
            self.client = MongoClient(mongostring)
            self.db = self.client.get_database()
        except Exception as e:
            logger.error('Cannot connect to database, did you set MONGO_GEO_STRING env variable?')
            raise
        self.collection = self.db[collection]
        self.stopwords = utils.getStopWordDict()
        self.terminosComunes = utils.getTerminosComunesDominioDict()
        self.terminosBusqueda = utils.getTerminosBusquedaDict()


    def findInDatabase(self,text):
        return self.collection.find({'$text': {'$search': text }},{'score': {'$meta': "textScore"}}).sort([('score',{'$meta': 'textScore'})])

    def cleanText(self,text):
        return utils.getTokensNoUserNoHashtag(utils.strip_accents(text))

    def removeTerms(self,text):
        ok_tokens = []
        for token in text.split():
            #Lower
            token = token.lower()
            #Stopwords,Terminos Busqueda,Terminos Comunes
            if not self.terminosBusqueda.get(token,False) and not self.terminosComunes.get(token,False) and not self.stopwords.get(token,False):
                ok_tokens.append(token)
        return ok_tokens

    def findSolutions(self,elements,solutions):
        self.findSelfSolutions(elements,solutions)
        return False


    def process(self,text,elements):
        #### explorar fuzzy text search e indices mongo...
        #ojo falta limpiar tildes etc
        #preprocess
        text = ' '.join(self.cleanText(text))
        #text = self.removeTerms(text)

        #trigrams
        text = self.processNgrams(text,elements,3)
        #bigrams
        text = self.processNgrams(text,elements,2)
        #unigrams
        text = ' '.join(self.removeTerms(text))
        text = self.processNgrams(text,elements,1)
        

    def processNgrams(self,text,elements,n):
        """
        Busco ngrama y si trae resultados saco la frase del texto para proximas busquedas.
        """
        ngrams_list = ngrams(text.split(), n)
        for ngram in ngrams_list:
            join_ngram = ' '.join(str(i) for i in ngram)
            double_quoted_ngram = '\"'+join_ngram+'\"'
            count = self.processText(double_quoted_ngram,elements)
            if count > 0:
                logger.debug(str(count)+' elementos '+self.__class__.__name__+' para '+str(double_quoted_ngram))
                text = text.replace(join_ngram,'')
        return text





    def processText(self,text,elements):
        shapes_found = []
        count = 0
        for element_found in self.findInDatabase(text):
            element_returned = {}
            element_returned[u'token'] = text.replace('\"','')
            element_returned[u'used'] = False
            element_returned[u'score_mongo'] = element_found['score']
            element_returned[u'score_ngram'] = len(element_returned[u'token'].split())
            element_returned[u'geo_type'] = element_found[u'geometry'][u'type']
            element_returned[u'geometry'] = element_found[u'geometry']
            element_returned[u'coll_type'] = self.__class__.__name__ 
            element_returned = self.transformParticulars(element_found,element_returned)
            element_returned[u'key'] = str(element_found[u'_id'])
            this_shape = asShape(element_found['geometry'])
            this_shape_found_before = False
            for shape in shapes_found:
                this_shape_found_before = this_shape.almost_equals(shape)
                if this_shape_found_before:
                    #print 'DUPLICATE FOUND'
                    break
            if not this_shape_found_before:
                shapes_found.append(this_shape)
                if not elements.get(element_returned[u'geo_type'],False):
                    elements[element_returned[u'geo_type']] = {}
                elements[element_returned[u'geo_type']][element_returned['key']] = element_returned
                count += 1
        return count

    def elementNearGeom(self,geom,id_element,distance):
        return self.collection.find_one({'_id':ObjectId(id_element),'geometry':{'$near':{'$geometry':geom,'$maxDistance': distance}}})
    
    def elementIntersectsGeom(self,geom,id_element):
        return self.collection.find_one({'_id':ObjectId(id_element),'geometry':{'$geoIntersects':{'$geometry':geom}}})


class GeoBarrio(GeoCollection):
    def transformParticulars(self,element_found,element_returned):
        element_returned[u'nombre'] = element_found[u'properties'][u'BARRIO']
        element_returned[u'codigo'] = element_found[u'properties'][u'CODBA']
        element_returned[u'allows_intersection'] = False
        element_returned[u'allows_lonely_solution'] = True
        return element_returned

class GeoCalle(GeoCollection):
    def transformParticulars(self,element_found,element_returned):
        element_returned[u'nombre'] = element_found['properties']['NOM_CALLE']
        element_returned[u'codigo'] = element_found['properties']['COD_NOMBRE']
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = False
        return element_returned

class GeoLugar(GeoCollection):
    def transformParticulars(self,element_found,element_returned):
        element_returned[u'nombre'] = element_found[u'properties'][u'NOMBRE']
        element_returned[u'codigo'] = '-'.join(element_found['properties']['NOMBRE'].split())
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = True
        return element_returned

class GeoEspacioLibre(GeoCollection):
    def transformParticulars(self,element_found,element_returned):
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = True
        if element_found[u'properties'][u'NOMBRE_ESP']:
            element_returned['nombre'] = element_found[u'properties'][u'NOM_TIPO_E'] + ' - ' + element_found[u'properties'][u'NOMBRE_ESP']
            element_returned[u'codigo'] = element_found[u'properties'][u'COD_NOM_ES']
        elif element_found[u'properties'][u'NOM_PARQUE']:
            element_returned['nombre'] = element_found[u'properties'][u'NOM_PARQUE']
            element_returned[u'codigo'] = element_found[u'properties'][u'COD_NOM_PA']
        return element_returned