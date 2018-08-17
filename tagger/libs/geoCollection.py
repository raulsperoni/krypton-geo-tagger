#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk import ngrams
from pymongo import MongoClient
from shapely.geometry import asShape
import pymongo
import utils
import logging
import spacy
import textacy

logger = logging.getLogger(__name__)


class GeoCollection(object):
    """
    Objeto generico del espacio.
    """

    def __init__(self, mongostring, collection, keyword_filename=None):
        try:
            self.client = MongoClient(mongostring)
            self.db = self.client.get_database()
        except Exception as e:
            logger.error('Cannot connect to database, did you set MONGO_GEO_STRING env variable?')
            raise
        self.collection = self.db[collection]
        self.keywords = None
        if keyword_filename:
            self.keywords = [
                textacy.preprocess_text(line, fix_unicode=True, no_punct=True, no_accents=True, transliterate=True) for
                line in open(keyword_filename, "r", encoding='utf-8')]
        self.stopwords = utils.getStopWordDict()
        self.terminosComunes = utils.getTerminosComunesDominioDict()
        self.terminosBusqueda = utils.getTerminosBusquedaDict()
        self.nlp = spacy.load('es')

    def getKeywords(self):
        return self.keywords

    def findInDatabase(self, text):
        return self.collection.find({'$text': {'$search': text}}, {'score': {'$meta': "textScore"}}).sort(
            [('score', {'$meta': 'textScore'})])

    def cleanText(self, text):
        return utils.getTokensNoUserNoHashtag(utils.strip_accents(text))

    def removeTerms(self, text):
        ok_tokens = []
        for token in text.split():
            # Lower
            token = token.lower()
            # Stopwords,Terminos Busqueda,Terminos Comunes
            if not self.terminosBusqueda.get(token, False) and not self.terminosComunes.get(token,
                                                                                            False) and not self.stopwords.get(
                    token, False):
                ok_tokens.append(token)
        return ok_tokens

    def findSolutions(self, elements, solutions):
        self.findSelfSolutions(elements, solutions)
        return False

    def process(self, doc, elements, list_of_entity_types_to_ignore=[]):
        """
        Entra un Doc, preprocesado, limpio y con entidades marcadas
        :param doc: Spacy Doc element
        :param elements: Georaphic elements found
        :return:
        """

        #### explorar fuzzy text search e indices mongo...
        # ojo falta limpiar tildes etc
        # preprocess
        #text = ' '.join(self.cleanText(text))
        # text = self.removeTerms(text)
        self.processEntities(doc,elements,list_of_entity_types_to_ignore)
        # trigrams
        self.processNgrams(doc, elements, 3)
        # bigrams
        self.processNgrams(doc, elements, 2)
        # unigrams
        #text = ' '.join(self.removeTerms(text))
        #text = self.processNgrams(text, elements, 1)
        #logger.info(text)

    def processEntities(self, doc, elements,list_of_entity_types_to_ignore):
        """

        :param doc: Spacy Doc element
        :param elements:
        :return:
        """
        for entity in doc.ents:
            # Solo proceso la entidad asociada a la Collection
            logger.debug('? %s,%s',self.__class__.__name__.upper(),entity.label_.upper())
            if entity.label_.upper() == self.__class__.__name__.upper():
                logger.debug('%s %s', 'Busco para entidad:', entity.text)
                double_quoted_entity = '\"' + entity.text + '\"'
                count = self.processText(double_quoted_entity, elements)
                if count > 0:
                    logger.debug('%s elementos %s para %s',str(count),self.__class__.__name__ ,str(double_quoted_entity))
                    # Marco los tokens involucrados para no volverlos a usar
                    for token in entity:
                        token._.set('with_results', True)
            elif entity.label_.upper() in [e.upper() for e in list_of_entity_types_to_ignore]:
                # Marco estos tokens pq no interesan
                for token in entity:
                    token._.set('with_results', True)
        return doc



    def processNgrams(self, doc, elements, n):
        """
        Busco ngrama y si trae marco los tokens del texto para proximas busquedas.
        """
        ngrams_list = ngrams(doc.text.split(), n)
        ngram_ini_token = 0
        ngram_end_token = n
        for ngram in ngrams_list:
            join_ngram = ' '.join(str(i) for i in ngram)
            span = doc[ngram_ini_token:ngram_end_token]
            if span and all([not token.ent_type and not token._.with_results for token in span]):
                assert span.text == join_ngram
                logger.debug('%s=%s %s','Busco para ngrama n',n,span)
                # Ningun token del ngrama tiene resultados anteriores
                double_quoted_ngram = '\"' + join_ngram + '\"'
                count = self.processText(double_quoted_ngram, elements)
                if count > 0:
                    logger.debug(str(count) + ' elementos ' + self.__class__.__name__ + ' para ' + str(double_quoted_ngram))
                    for token in span:
                        token._.with_results = True
            ngram_ini_token += 1
            ngram_end_token += 1
        return doc

    def processText(self, text, elements):
        shapes_found = []
        count = 0
        for element_found in self.findInDatabase(text):
            element_returned = {}
            element_returned[u'token'] = text.replace('\"', '')
            element_returned[u'used'] = False
            element_returned[u'score_mongo'] = element_found['score']
            element_returned[u'score_ngram'] = len(element_returned[u'token'].split())
            element_returned[u'geo_type'] = element_found[u'geometry'][u'type']
            element_returned[u'geometry'] = element_found[u'geometry']
            element_returned[u'coll_type'] = self.__class__.__name__
            element_returned = self.transformParticulars(element_found, element_returned)
            element_returned[u'key'] = str(element_found[u'_id'])
            this_shape = asShape(element_found['geometry'])
            this_shape_found_before = False
            for shape in shapes_found:
                this_shape_found_before = this_shape.almost_equals(shape)
                if this_shape_found_before:
                    # print 'DUPLICATE FOUND'
                    break
            if not this_shape_found_before:
                shapes_found.append(this_shape)
                if not elements.get(element_returned[u'geo_type'], False):
                    elements[element_returned[u'geo_type']] = {}
                elements[element_returned[u'geo_type']][element_returned['key']] = element_returned
                count += 1
        return count

    def elementNearGeom(self, geom, id_element, distance):
        return self.collection.find_one(
            {'_id': ObjectId(id_element), 'geometry': {'$near': {'$geometry': geom, '$maxDistance': distance}}})

    def elementIntersectsGeom(self, geom, id_element):
        return self.collection.find_one(
            {'_id': ObjectId(id_element), 'geometry': {'$geoIntersects': {'$geometry': geom}}})


class GeoBarrio(GeoCollection):
    def transformParticulars(self, element_found, element_returned):
        element_returned[u'nombre'] = element_found[u'properties'][u'BARRIO']
        element_returned[u'codigo'] = element_found[u'properties'][u'CODBA']
        element_returned[u'allows_intersection'] = False
        element_returned[u'allows_lonely_solution'] = True
        return element_returned


class GeoCalle(GeoCollection):

    def transformParticulars(self, element_found, element_returned):
        element_returned[u'nombre'] = element_found['properties']['NOM_CALLE']
        element_returned[u'codigo'] = element_found['properties']['COD_NOMBRE']
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = False
        return element_returned


class GeoLugar(GeoCollection):

    def transformParticulars(self, element_found, element_returned):
        element_returned[u'nombre'] = element_found[u'properties'][u'NOMBRE']
        element_returned[u'codigo'] = '-'.join(element_found['properties']['NOMBRE'].split())
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = True
        return element_returned


class GeoEspacioLibre(GeoCollection):

    def transformParticulars(self, element_found, element_returned):
        element_returned[u'allows_intersection'] = True
        element_returned[u'allows_lonely_solution'] = True
        if element_found[u'properties'][u'NOMBRE_ESP']:
            element_returned['nombre'] = element_found[u'properties'][u'NOM_TIPO_E'] + ' - ' + \
                                         element_found[u'properties'][u'NOMBRE_ESP']
            element_returned[u'codigo'] = element_found[u'properties'][u'COD_NOM_ES']
        elif element_found[u'properties'][u'NOM_PARQUE']:
            element_returned['nombre'] = element_found[u'properties'][u'NOM_PARQUE']
            element_returned[u'codigo'] = element_found[u'properties'][u'COD_NOM_PA']
        return element_returned
