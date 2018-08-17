#!/usr/bin/env python
# -*- coding: utf-8 -*-
import utils
from geoCollection import GeoCalle,GeoEspacioLibre,GeoLugar,GeoBarrio
from pprint import pprint
import functools,time,sys,traceback
from shapely.geometry import asShape,MultiPoint,box
import logging
import spacy
from spacy.tokens import Token
import textacy
from spacy_lookup import Entity
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GeoEngine(object):
    """
    Clase para hacer enrich con ubicaciones a partir de texto
    """
    def __init__(self,mongostring):
        self.decimals_solutions_equals = 3
        self.near_distance_limit = 0.0005
        city_box = utils.getCityBoundingBox()
        self.city_limits = box(city_box[0],city_box[1],city_box[2],city_box[3])
        try:
            self.geo_elements = [
                GeoCalle(mongostring, 'v_mdg_vias', keyword_filename='calles.txt'),
                GeoLugar(mongostring,'uptu'),
                GeoEspacioLibre(mongostring,'v_mdg_espacios_libres'),
                GeoBarrio(mongostring,'limites_barrios')
            ]
        except Exception as e:
            logger.error(e)
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

        #Spacy NLP
        self.nlp = spacy.load('es')

        #Generic entity types
        self.city_kewords = [textacy.preprocess_text(line, fix_unicode=True,no_punct=True,no_accents=True,transliterate=True) for line in open('palabras_clave_ciudad.txt',"r",encoding='utf-8')]
        self.domain_kewords = [textacy.preprocess_text(line, fix_unicode=True,no_punct=True,no_accents=True,transliterate=True) for line in open('palabras_clave_dominio.txt',"r",encoding='utf-8')]
        entity_domain= Entity(keywords_list=self.domain_kewords,label="KEY_DOMAIN")
        entity_city = Entity(keywords_list=self.city_kewords, label="KEY_CITY")
        self.nlp.add_pipe(entity_domain,name='KEY_DOMAIN', last=True)
        self.nlp.add_pipe(entity_city,name='KEY_CITY', last=True)
        self.list_of_entity_types_to_ignore_in_search = ['KEY_CITY','KEY_DOMAIN']

        #Geo collection specific entity types
        for geo_element in self.geo_elements:
            if geo_element.getKeywords():
                entity_type = Entity(keywords_list=geo_element.getKeywords(), label=geo_element.__class__.__name__.upper())
                self.nlp.add_pipe(entity_type,name=geo_element.__class__.__name__,last=True)

        #Add custom attribute to Spacy tokens
        Token.set_extension('with_results', default=False)
        Token.set_extension('part_of_solution', default=False)


    def timeit(func):
        @functools.wraps(func)
        def newfunc(*args, **kwargs):
            startTime = time.time()
            result = func(*args, **kwargs)
            elapsedTime = time.time() - startTime
            logger.debug('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
            return result
        return newfunc

    def addGeoCollection(self,geo_collection):
        self.geo_elements += geo_element

    @timeit
    def preProcess(self,text):
        text = textacy.preprocess_text(text, fix_unicode=True,lowercase=False, no_punct=True,no_urls=True,no_emails=True,no_accents=True,transliterate=True)
        textacy.preprocess.normalize_whitespace(text)
        logger.debug('%s,%s',"POST_PREPROCESS >",text)
        return text

    @timeit
    def markTextWithKnownStuff(self,text):
        doc = self.nlp(text)
        for ent in doc.ents:
            logger.debug('%s,%s,%s,%s',ent.text, ent.start_char, ent.end_char, ent.label_)
        return doc


    #@timeit
    def process(self,text,coordinates):
        found_solutions = []
        logger.info('Entrada <%s> %s',text,coordinates)

        # Existing coordinates win
        if coordinates:
            entry_coords = asShape(coordinates)
            if entry_coords.intersects(self.city_box):
                solution = self.buildSolution([],entry_coords)
                solution['score']['combined'] = 100
                solutions.append(solution)
                # Im done
                return found_solutions

        #Preprocess text with nlp features
        pp_text = self.preProcess(text)
        logger.debug(pp_text)
        doc = self.markTextWithKnownStuff(pp_text)

        # Starting the search
        found_elements = {}
        ## For each geoElement in list find process it with tokens
        for coll in self.geo_elements:
            coll.process(doc,found_elements,self.list_of_entity_types_to_ignore_in_search)
        self.findSolutions(found_elements,found_solutions)

        ## Sort, filter and return.
        found_solutions.sort(key=lambda solucion: solucion['score']['combined'], reverse=True)
        for s in found_solutions[:1]:
            self.printSolution(s)

        #logger.info(found_solutions)
        return found_solutions

    
    def findSolutions(self,elements,solutions):
        #dentro de un mismo geoTipo
        self.findSameTypeSolutions(elements,solutions)
        #intersecciones o near de diferentes tipos
        self.findCrossTypeSolutions(elements,solutions)
        #soluciones de un solo elemento
        self.findLonelySolutions(elements,solutions)

    @timeit
    def findLonelySolutions(self,elements,solutions):
        shapes_found = []
        logger.debug('Buscando soluciones de un solo elemento...')
        for geo_type in elements.keys():
            for element_id_a in elements[geo_type].keys():
                element_a = elements[geo_type][element_id_a]
                if element_a['allows_lonely_solution']:
                    shape_a = asShape(element_a['geometry'])
                    middle_point = shape_a.representative_point()
                    shape_found_before = False
                    for shape in shapes_found:
                        shape_found_before = middle_point.almost_equals(shape,self.decimals_solutions_equals)
                        if shape_found_before:
                            #print 'DUPLICATE FOUND'
                            break
                    if not shape_found_before:
                        shapes_found.append(middle_point)
                        solution = self.buildSolution([element_a],middle_point)
                        solutions.append(solution)
                        #self.printSolution(solution)



    @timeit
    def findCrossTypeSolutions(self,elements,solutions):
        shapes_found = []
        logger.debug('Buscando soluciones cruzadas...')
        for geo_type_a in elements.keys():
            for geo_type_b in elements.keys():
                if geo_type_a != geo_type_b:
                    for element_id_a in elements[geo_type_a].keys():
                        for element_id_b in elements[geo_type_b].keys():
                            element_a = elements[geo_type_a][element_id_a]
                            element_b = elements[geo_type_b][element_id_b]
                            if not element_a['used'] and not element_b['used'] and element_a['allows_intersection'] and element_b['allows_intersection'] and not self.shareToken(element_a,element_b):
                                shape_a = asShape(element_a['geometry'])
                                shape_b = asShape(element_b['geometry'])
                                middle_point = None
                                #print element_a['nombre'],element_b['nombre'], shape_a.distance(shape_b), self.near_distance_limit
                                if shape_a.crosses(shape_b):
                                    middle_point = shape_a.intersection(shape_b).representative_point()
                                elif shape_a.distance(shape_b) > 0 and shape_a.distance(shape_b) < self.near_distance_limit:
                                    middle_point = MultiPoint([shape_a.representative_point(),shape_b.representative_point()]).representative_point()
                                if middle_point:
                                    shape_found_before = False
                                    for shape in shapes_found:
                                        shape_found_before = middle_point.almost_equals(shape,self.decimals_solutions_equals)
                                        if shape_found_before:
                                            #print 'DUPLICATE FOUND'
                                            break
                                    if not shape_found_before:
                                        shapes_found.append(middle_point)
                                        #elements[geo_type_a][element_id_a]['used'] = True
                                        #elements[geo_type_b][element_id_b]['used'] = True
                                        solution = self.buildSolution([element_a,element_b],middle_point)
                                        solutions.append(solution)
                                        #self.printSolution(solution)

    @timeit
    def findSameTypeSolutions(self,elements,solutions):
        shapes_found = []
        for geo_type in elements.keys():
            logger.debug('Buscando soluciones para '+geo_type+'...')
            for element_id_a in elements[geo_type].keys():
                for element_id_b in elements[geo_type].keys():
                    if element_id_a != element_id_b:
                        element_a = elements[geo_type][element_id_a]
                        element_b = elements[geo_type][element_id_b]
                        #No quiero comparar los mismos elementos
                        #No quiero comparar elementos encontrados con tokens similares
                        if element_a['codigo'] != element_b['codigo'] and not self.shareToken(element_a,element_b):
                            shape_a = asShape(element_a['geometry'])
                            shape_b = asShape(element_b['geometry'])
                            #Estoy buscando que se interecten
                            if shape_a.intersects(shape_b):
                                #La referencia para la solucion es este punto
                                intersection = shape_a.intersection(shape_b).representative_point()
                                shape_found_before = False
                                for shape in shapes_found:
                                    #Si los puntos son similares estoy frente a una solucion duplicada
                                    shape_found_before = intersection.almost_equals(shape,self.decimals_solutions_equals)
                                    if shape_found_before:
                                        #print 'DUPLICATE FOUND'
                                        break
                                #Si el punto de referencia no fue encontrado antes armo una solucion
                                if not shape_found_before:
                                    shapes_found.append(intersection)
                                    element_a['used'] = True
                                    element_b['used'] = True
                                    solution = self.buildSolution([element_a,element_b],intersection)
                                    solutions.append(solution)
                                    #self.printSolution(solution)

    def buildSolution(self,elements,intersection):
        score = {}
        score['mongo'] = sum(e['score_mongo'] for e in elements)
        score['ngram'] = sum(e['score_ngram'] for e in elements)
        score['count'] = len(elements)
        score['combined'] = score['mongo']*score['ngram']*score['count']
        return {'centroid':list(intersection.coords),'elements': elements,'score':score}

    def printSolution(self,solution):
        logger.info('   Solucion >> '+str(solution['centroid'])+'\n')
        for e in solution['elements']:
            logger.info('       '+str(e['nombre'].encode('utf8'))+'\n')
        logger.info('       Score > %s',solution['score'])

    def shareToken(self, element_a,element_b):
        return element_a['token'] in element_b['token'] or element_b['token'] in element_a['token']
