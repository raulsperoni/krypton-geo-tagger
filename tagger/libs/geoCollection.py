#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk import ngrams
from pymongo import MongoClient
from shapely.geometry import asShape
import functools,time
import logging
import spacy
from spacy.tokens import Span
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
        self.nlp = spacy.load('es')

    def timeit(func):
        @functools.wraps(func)
        def newfunc(*args, **kwargs):
            startTime = time.time()
            result = func(*args, **kwargs)
            elapsedTime = time.time() - startTime
            logger.debug('function [{}], finished in {} ms'.format(func.__name__,int(elapsedTime * 1000)))
            return result
        return newfunc

    def getKeywords(self):
        return self.keywords

    def findInDatabase(self, text):
        double_quoted_text = '\"' + text + '\"'
        return self.collection.find({'$text': {'$search': double_quoted_text}}, {'score': {'$meta': "textScore"}}).sort(
            [('score', {'$meta': 'textScore'})])

    def findSolutions(self, elements, solutions):
        self.findSelfSolutions(elements, solutions)
        return False

    @timeit
    def process(self, doc, elements, entity_types_to_ignore,entity_types_to_process):
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
        self.processEntities(doc,elements,entity_types_to_ignore,entity_types_to_process)
        # trigrams
        self.processNgrams(doc, elements, 3)
        # bigrams
        self.processNgrams(doc, elements, 2)
        # unigrams without stopwords or pron or conj
        #unigrams = [token for token in doc if not token.is_stop and not token._.with_results and token.pos_ not in ['PRON','CONJ']]
        self.processNgrams(doc, elements, 1)

    #@timeit
    def processEntities(self, doc, elements,entity_types_to_ignore,entity_types_to_process):
        """

        :param doc: Spacy Doc element
        :param elements:
        :return:
        """
        #logger.debug('processEntities in %s, ignoring %s',self.__class__.__name__.upper(),list_of_entity_types_to_ignore)
        for entity in doc.ents:
            # Ignoro entidades asociadas a keywords
            if entity.label_.upper() in [e.upper() for e in entity_types_to_ignore]:
                # Marco estos tokens pq no interesan
                for token in entity:
                    token._.set('with_results', True)
            # Solo proceso la entidad asociada a la Collection
            #logger.debug('? %s,%s',self.__class__.__name__.upper(),entity.label_.upper())
            #TODO: este if tiene sentido? caso que encuentra una calle como entidad persona no entra.
            if entity.label_.upper() in entity_types_to_process:
                #logger.debug('%s %s', 'Busco para entidad:', entity.text)
                self.processText(entity, elements)
        return doc

    #@timeit
    def processNgrams(self, doc, elements, n):
        """
        Busco ngrama y si trae marco los tokens del texto para proximas busquedas .
        """
        #logger.debug('processNgrams in %s, n=%s',self.__class__.__name__.upper(),n)
        ngrams_list = ngrams(doc, n)
        ngram_ini_token = 0
        ngram_end_token = n
        for ngram in ngrams_list:
            span = doc[ngram_ini_token:ngram_end_token]
            #logger.debug('span vector %s',tokens.vector)
            #Que haya tokens sin usar en resultados anteriores
            if len(span)>0 and all([not token._.with_results for token in span]):
                #logger.debug('%s=%s %s','Busco para ngrama n',n,text)
                self.processText(span, elements)
            ngram_ini_token += 1
            ngram_end_token += 1
        return doc

    #@timeit
    def processText(self, span, elements):
        """

        :param tokens: a Spacy Span: list of Tokens.
        :param elements:
        :return:
        """
        assert 'Span' == span.__class__.__name__
        #shapes_found = []
        count = 0
        for element_found in self.findInDatabase(span.text):
            element_returned = {}
            element_returned['vector'] = span.vector_norm
            element_returned['token_start_char'] = span.start_char
            element_returned['token_end_char'] = span.end_char
            element_returned[u'token'] = span.text
            element_returned[u'times_used_in_solution'] = 0
            element_returned[u'score_mongo'] = element_found['score']
            element_returned[u'score_ngram'] = len(span)
            element_returned[u'geo_type'] = element_found[u'geometry'][u'type']
            element_returned[u'geometry'] = element_found[u'geometry']
            element_returned[u'coll_type'] = self.__class__.__name__
            element_returned = self.transformParticulars(element_found, element_returned)
            element_returned[u'key'] = str(element_found[u'_id'])
            #this_shape = asShape(element_found['geometry'])
            #this_shape_found_before = False
            #for shape in shapes_found:
            #    this_shape_found_before = this_shape.almost_equals(shape)
            #    if this_shape_found_before:
            #        logger.debug('DUPLICATE FOUND')
            #        break
            #if not this_shape_found_before:
            #shapes_found.append(this_shape)
            if not elements.get(element_returned[u'geo_type'], False):
                elements[element_returned[u'geo_type']] = {}
            elements[element_returned[u'geo_type']][element_returned['key']] = element_returned
            count += 1
        if count>0:
            # Marco tokens como usados
            for token in span:
                token._.with_results = True
            logger.debug('%s elementos %s para %s', count, self.__class__.__name__, span.text)
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
