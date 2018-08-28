#!/usr/bin/env python
# -*- coding: utf-8 -*-
from geoCollection import GeoCalle, GeoEspacioLibre, GeoLugar, GeoBarrio
import functools, time, sys, traceback
from shapely.geometry import asShape, MultiPoint, box
import logging, os
import spacy
from spacy.tokens import Token
import textacy
from spacy_lookup import Entity
import spacy_lookup
import re
from entityMatcher import EntityMatcher

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GeoEngine(object):
    """
    Clase para hacer enrich con ubicaciones a partir de texto
    """

    def __init__(self, mongostring):
        self.decimals_solutions_equals = 3
        self.near_distance_limit = 0.0005
        city_box = self.getCityBoundingBox()
        self.city_limits = box(city_box[0], city_box[1], city_box[2], city_box[3])
        try:
            self.geo_elements = [
                GeoCalle(mongostring, 'v_mdg_vias', keyword_filename='calles.txt'),
                GeoLugar(mongostring, 'uptu', keyword_filename='uptu.txt'),
                GeoEspacioLibre(mongostring, 'v_mdg_espacios_libres'),
                GeoBarrio(mongostring, 'limites_barrios',keyword_filename='limites_barrios.txt')
            ]
        except Exception as e:
            logger.error(e)
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

        # Spacy NLP
        self.nlp = spacy.load('es')

        self.hashtag_re = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
        self.mention_re = re.compile("(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", re.UNICODE)

        keywordlists = [
            ('palabras_clave_ciudad.txt','KEY_CITY'),
            ('palabras_clave_dominio.txt','KEY_DOMAIN'),
            ('organizaciones.txt','KEY_ORG'),
            ('personas.txt','KEY_PER'),
            ('ignore.txt','KEY_IGNORE')
        ]
        # Generic entity types
        for (file,label) in keywordlists:
            keywords = [
                textacy.preprocess_text(line, fix_unicode=True, no_punct=False, no_accents=True, transliterate=True) for line
                in open(file, "r", encoding='utf-8')]
            entity = Entity(keywords_list=keywords, label=label)
            self.nlp.add_pipe(entity, name=label, after='ner')


        self.list_of_entity_types_to_ignore_in_search = [label for (file,label) in keywordlists]
        self.list_of_entity_types_to_process_in_search = ['PER','LOC','ORG','MISC']

        # Geo collection specific entity types
        for geo_element in self.geo_elements:
            if geo_element.getKeywords():
                entity_type = Entity(keywords_list=geo_element.getKeywords(),label=geo_element.__class__.__name__.upper())
                self.nlp.add_pipe(entity_type, name=geo_element.__class__.__name__, after='ner')
                self.list_of_entity_types_to_process_in_search.append(geo_element.__class__.__name__.upper())



        # Add custom attribute to Spacy tokens
        Token.set_extension('ignore', default=False)
        Token.set_extension('with_results', default=False)
        Token.set_extension('part_of_same_type_solution', default=False)
        Token.set_extension('part_of_cross_type_solution', default=False)
        Token.set_extension('part_of_lonely_solution', default=False)


    def getCityBoundingBox(self):
        """
        """
        here = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(here, 'city_bounding_box.txt')
        bounding_box = open(filename, encoding='utf-8').read()
        result = []
        for coord in bounding_box.split(','):
            result.append(float(coord))
        return result

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

    def addGeoCollection(self, geo_collection):
        self.geo_elements += geo_element

    @timeit
    def preProcess(self, text):
        text = self.hashtag_re.sub('*HASHTAG*', text)
        text = self.mention_re.sub('*MENTION*', text)
        text = textacy.preprocess_text(text, fix_unicode=True, lowercase=False, no_punct=True, no_urls=True,
                                       no_emails=True, no_accents=True, transliterate=True)
        textacy.preprocess.normalize_whitespace(text)
        logger.debug('%s,%s', "POST_PREPROCESS >", text)
        return text

    @timeit
    def markTextWithKnownStuff(self, text,ents_to_ignore):
        try:
            doc = self.nlp(text)
        except Exception as e:
            logger.error('Error, deshabilitando entity pipe GeoLugar')
            disabled = self.nlp.disable_pipes('GeoLugar')
            doc = self.nlp(text)
            disabled.restore()
            logger.error('Rehabilitando GeoLugar')

        for ent in doc.ents:
            ignored = False
            if ent.label_ in ents_to_ignore:
                ignored = True
                for token in ent:
                    token._.ignore = True
            logger.debug('%s [%s,%s] -> %s,%s', ent.text, ent.start_char, ent.end_char, ent.label_,ignored)
        return doc

    @timeit
    def process(self, text):
        found_solutions = []
        logger.info('Entrada <%s>', text)

        #ideas>
        #regex nombres de usuario, hashtags https://spacy.io/api/matcher#add

        # Preprocess text with nlp features
        pp_text = self.preProcess(text)
        doc = self.markTextWithKnownStuff(pp_text,self.list_of_entity_types_to_ignore_in_search)

        # Starting the search
        found_elements = {}
        solution_index = {}
        for coll in self.geo_elements:
            coll.process(doc, found_elements, self.list_of_entity_types_to_ignore_in_search,self.list_of_entity_types_to_process_in_search)
            self.findSameTypeSolutions(doc, found_elements, found_solutions, solution_index)


        self.findCrossTypeSolutions(doc, found_elements, found_solutions, solution_index)

        self.findLonelySolutions(doc, found_elements, found_solutions, solution_index)

        ## Sort, filter and return.
        found_solutions.sort(key=lambda solucion: solucion['score']['combined'], reverse=True)
        for s in found_solutions[:1]:
            self.printSolution(s)

        # logger.info(found_solutions)
        return found_solutions

    @timeit
    def findLonelySolutions(self, doc, elements, solutions, solution_index):
        shapes_found = []
        logger.debug('Buscando soluciones de un solo elemento...')
        for geo_type in elements.keys():
            for element_id_a in elements[geo_type].keys():
                element_a = elements[geo_type][element_id_a]
                span_a = doc.char_span(element_a['token_start_char'], element_a['token_end_char'])
                if element_a['allows_lonely_solution'] and all(
                        [not token._.part_of_cross_type_solution and not token._.part_of_same_type_solution for token in
                         span_a]):
                    shape_a = asShape(element_a['geometry'])
                    middle_point = shape_a.representative_point()
                    shape_found_before = False
                    for shape in shapes_found:
                        shape_found_before = middle_point.almost_equals(shape, self.decimals_solutions_equals)
                        if shape_found_before:
                            logger.debug('DUPLICATE FOUND')
                            break
                    if not shape_found_before:
                        shapes_found.append(middle_point)
                        solutions.append(self.buildSolution(doc, [element_a], middle_point, solution_index))

    @timeit
    def findCrossTypeSolutions(self, doc, elements, solutions, solution_index):
        shapes_found = []
        logger.debug('Buscando soluciones cruzadas...')
        for geo_type_a in elements.keys():
            for geo_type_b in elements.keys():
                if geo_type_a != geo_type_b:

                    for element_id_a in elements[geo_type_a].keys():
                        element_a = elements[geo_type_a][element_id_a]
                        span_a = doc.char_span(element_a['token_start_char'], element_a['token_end_char'])

                        for element_id_b in elements[geo_type_b].keys():
                            element_b = elements[geo_type_b][element_id_b]
                            span_b = doc.char_span(element_b['token_start_char'], element_b['token_end_char'])

                            if element_id_a != element_id_b and element_a['vector'] != element_b[
                                'vector'] and not self.inSolutionIndex(solution_index, element_a['vector'],
                                                                       element_b['vector']):
                                # logger.debug('%s [%s,%s] # %s [%s,%s]',span_a,span_a.start,span_a.end,span_b,span_b.start,span_b.end)
                                # No quiero un solucion que ya existe a nivel de tokens

                                if element_a['allows_intersection'] and element_b['allows_intersection'] \
                                        and all([not token._.part_of_same_type_solution for token in span_a]) \
                                        and all([not token._.part_of_same_type_solution for token in span_b]):
                                    shape_a = asShape(element_a['geometry'])
                                    shape_b = asShape(element_b['geometry'])
                                    middle_point = None
                                    # print element_a['nombre'],element_b['nombre'], shape_a.distance(shape_b), self.near_distance_limit
                                    if shape_a.crosses(shape_b):
                                        middle_point = shape_a.intersection(shape_b).representative_point()
                                    elif shape_a.distance(shape_b) > 0 and shape_a.distance(
                                            shape_b) < self.near_distance_limit:
                                        middle_point = MultiPoint([shape_a.representative_point(),
                                                                   shape_b.representative_point()]).representative_point()
                                    if middle_point:
                                        shape_found_before = False
                                        for shape in shapes_found:
                                            shape_found_before = middle_point.almost_equals(shape,
                                                                                            self.decimals_solutions_equals)
                                            if shape_found_before:
                                                logger.debug('DUPLICATE FOUND')
                                                break
                                        if not shape_found_before:
                                            shapes_found.append(middle_point)
                                            solutions.append(
                                                self.buildSolution(doc, [element_a, element_b], middle_point,
                                                                   solution_index))

    @timeit
    def findSameTypeSolutions(self, doc, elements, solutions, solution_index):
        shapes_found = []
        for geo_type in elements.keys():
            logger.debug('Buscando soluciones para ' + geo_type + '...')

            for element_id_a in elements[geo_type].keys():
                element_a = elements[geo_type][element_id_a]
                span_a = doc.char_span(element_a['token_start_char'], element_a['token_end_char'])

                for element_id_b in elements[geo_type].keys():
                    element_b = elements[geo_type][element_id_b]
                    span_b = doc.char_span(element_b['token_start_char'], element_b['token_end_char'])

                    if element_id_a != element_id_b and element_a['vector'] != element_b[
                        'vector'] and not self.inSolutionIndex(solution_index, element_a['vector'],
                                                               element_b['vector']):
                        # logger.debug('%s [%s,%s] # %s [%s,%s]',span_a,span_a.start,span_a.end,span_b,span_b.start,span_b.end)
                        shape_a = asShape(element_a['geometry'])
                        shape_b = asShape(element_b['geometry'])
                        if shape_a.intersects(shape_b):
                            # La referencia para la solucion es este punto
                            intersection = shape_a.intersection(shape_b).representative_point()
                            if any([intersection.almost_equals(shape, self.decimals_solutions_equals) for shape in
                                    shapes_found]):
                                # ya encontre?
                                logger.debug('GEO DUPLICATE FOUND: %s, %s', span_a, span_b)
                            else:
                                # Si el punto de referencia no fue encontrado antes armo una solucion
                                shapes_found.append(intersection)
                                solutions.append(
                                    self.buildSolution(doc, [element_a, element_b], intersection, solution_index))

    def buildSolution(self, doc, elements, intersection, solution_index):
        score = {}
        score['mongo'] = sum(e['score_mongo'] for e in elements)
        score['ngram'] = sum(e['score_ngram'] for e in elements)
        score['count'] = len(elements)
        score['combined'] = score['mongo'] * score['ngram'] * score['count']
        solution = {'centroid': list(intersection.coords), 'elements': elements, 'score': score,
                    'show': '/'.join([element['nombre'] for element in elements])}

        tokens_lonely_solution = 0
        tokens_same_type_solution = 0
        tokens_cross_type_solution = 0

        if len(elements) == 1:
            # Lonely Solution
            element = elements[0]
            logger.debug('%s in [%s,%s] is now part of lonely solution', element['token'],
                         element['token_start_char'],
                         element['token_end_char'])
            original_span = doc.char_span(element['token_start_char'], element['token_end_char'])
            for token in original_span:
                token._.part_of_lonely_solution = True
                tokens_lonely_solution+=1
            self.storeInSolutionIndex(solution_index, element['vector'], element['vector'])
        elif elements.count(elements[0]) == len(elements):
            # Same type solution
            assert len(elements) == 2
            for element in elements:
                element['times_used_in_solution'] += 1
                logger.debug('%s in [%s,%s] is now part of same type solution', element['token'],
                             element['token_start_char'],
                             element['token_end_char'])
                original_span = doc.char_span(element['token_start_char'], element['token_end_char'])
                for token in original_span:
                    token._.part_of_same_type_solution = True
                    tokens_same_type_solution+=1
            self.storeInSolutionIndex(solution_index, elements[0]['vector'], elements[1]['vector'])
        else:
            # Cross type solution
            assert len(elements) == 2
            for element in elements:
                element['times_used_in_solution'] += 1
                logger.debug('%s in [%s,%s] is now part of cross type solution', element['token'],
                             element['token_start_char'],
                             element['token_end_char'])
                original_span = doc.char_span(element['token_start_char'], element['token_end_char'])
                for token in original_span:
                    token._.part_of_cross_type_solution = True
                    tokens_cross_type_solution+=1
            self.storeInSolutionIndex(solution_index, elements[0]['vector'], elements[1]['vector'])

        solution['score']['count_tokens_lonely_solution'] = tokens_lonely_solution
        solution['score']['count_same_type_solution'] = tokens_same_type_solution
        solution['score']['count_cross_type_solution'] = tokens_cross_type_solution

        return solution

    def printSolution(self, solution):
        logger.info('   Solucion >> ' + str(solution['centroid']) + '\n')
        for e in solution['elements']:
            logger.info('       ' + str(e['nombre'].encode('utf8')) + '\n')
        logger.info('       Score > %s', solution['score'])

    def shareToken(self, element_a, element_b):
        return element_a['token'] in element_b['token'] or element_b['token'] in element_a['token']

    def inSolutionIndex(self, solution_index, a, b):
        return (solution_index.get(a, False) and solution_index[a].get(b)) or (
                    solution_index.get(b, False) and solution_index[b].get(a))

    def storeInSolutionIndex(self, solution_index, a, b):
        solution_index[a] = solution_index.get(a, {})
        solution_index[a][b] = True
        solution_index[b] = solution_index.get(b, {})
        solution_index[b][a] = True
